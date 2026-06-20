"""Model-agnostic preprocessing adapter.

A single source of truth for how a classifier wants its inputs fed: target spatial
size, channel mean/std, and interpolation. Every classifier preprocessing in the
codebase reduces, after an optional resize, to one affine op ``(img - mean) / std``:

- ``[0,1] -> [-1,1]`` (the ViT/BEiT convention) is just ``mean = std = 0.5``
- ``"none"`` (feed [0,1] as-is) is ``mean = 0, std = 1``
- SDP-CROWN / ImageNet are other mean/std pairs

`resolve_preproc_spec` reads the real per-model values from the model itself
(HuggingFace ``AutoImageProcessor`` / timm ``pretrained_cfg`` / ONNX graph), with an
optional per-model ``override`` (e.g. from a hydra model yaml) winning over any field.

All callers pass images in ``[0,1]``. The denoiser path converts its ``[-1,1]`` output
to ``[0,1]`` (``* 0.5 + 0.5``) before calling :meth:`PreprocSpec.apply`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# SDP-CROWN CIFAR-10 normalization (matches the historical apply_pytorch_normalization).
SDPCROWN_MEAN: tuple[float, float, float] = (125.3 / 255, 123.0 / 255, 113.9 / 255)
SDPCROWN_STD: tuple[float, float, float] = (0.225, 0.225, 0.225)

# Inception / ViT-BEiT convention: [0,1] -> [-1,1].
HALF_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
HALF_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

# Identity: feed [0,1] unchanged.
IDENTITY_MEAN: tuple[float, float, float] = (0.0, 0.0, 0.0)
IDENTITY_STD: tuple[float, float, float] = (1.0, 1.0, 1.0)


def normalize(imgs: torch.Tensor, mean, std) -> torch.Tensor:
    """Apply ``(imgs - mean) / std``. Handles (C,H,W) and (B,C,H,W) tensors."""
    mean_t = torch.as_tensor(mean, device=imgs.device, dtype=imgs.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, device=imgs.device, dtype=imgs.dtype).view(-1, 1, 1)
    return (imgs - mean_t) / std_t


@dataclass(frozen=True)
class PreprocSpec:
    """How to turn a [0,1] image batch into a classifier-ready tensor."""

    target_size: tuple[int, int]  # (H, W)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bicubic"
    antialias: bool = True

    def resize(self, imgs01: torch.Tensor) -> torch.Tensor:
        return F.interpolate(imgs01, self.target_size, mode=self.interpolation, antialias=self.antialias)

    def normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        return normalize(imgs, self.mean, self.std)

    def apply(self, imgs01: torch.Tensor) -> torch.Tensor:
        """Resize then normalize. ``imgs01`` must be in [0,1] (B,C,H,W)."""
        return self.normalize(self.resize(imgs01))


def _as_dict(override) -> dict:
    if override is None:
        return {}
    if isinstance(override, Mapping):
        return dict(override)
    # OmegaConf DictConfig or similar
    try:
        from omegaconf import OmegaConf

        return dict(OmegaConf.to_container(override, resolve=True))
    except Exception:
        return dict(override)


def _triple(value) -> tuple[float, float, float]:
    seq = list(value)
    if len(seq) == 1:
        seq = seq * 3
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _hf_size(size) -> tuple[int, int]:
    if isinstance(size, Mapping):
        if "height" in size and "width" in size:
            return (int(size["height"]), int(size["width"]))
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return (edge, edge)
        # fall back to first numeric value
        vals = [int(v) for v in size.values() if isinstance(v, (int, float))]
        if vals:
            return (vals[0], vals[0])
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return (int(size[0]), int(size[1]))
    if isinstance(size, (int, float)):
        return (int(size), int(size))
    raise ValueError(f"Could not interpret HuggingFace processor size: {size!r}")


def _resolve_huggingface(classifier_name: str) -> PreprocSpec:
    from transformers import AutoImageProcessor

    try:
        proc = AutoImageProcessor.from_pretrained(classifier_name, local_files_only=True)
    except Exception:
        proc = AutoImageProcessor.from_pretrained(classifier_name)
    target_size = _hf_size(getattr(proc, "size", None))
    mean = _triple(getattr(proc, "image_mean", HALF_MEAN))
    std = _triple(getattr(proc, "image_std", HALF_STD))
    return PreprocSpec(target_size=target_size, mean=mean, std=std)


def _resolve_timm(classifier) -> PreprocSpec:
    cfg = getattr(classifier, "pretrained_cfg", None) or getattr(classifier, "default_cfg", None) or {}
    input_size = cfg.get("input_size", (3, 512, 512))
    target_size = (int(input_size[1]), int(input_size[2]))
    mean = _triple(cfg.get("mean", HALF_MEAN))
    std = _triple(cfg.get("std", HALF_STD))
    interpolation = cfg.get("interpolation", "bicubic")
    return PreprocSpec(target_size=target_size, mean=mean, std=std, interpolation=interpolation)


def resolve_preproc_spec(
    classifier,
    classifier_type: str,
    classifier_name: str | None = None,
    *,
    pytorch_normalization: str = "none",
    override=None,
) -> PreprocSpec:
    """Resolve a :class:`PreprocSpec` for ``classifier``.

    - ``huggingface``: read size/mean/std from ``AutoImageProcessor``.
    - ``timm``: read input_size/mean/std/interpolation from ``pretrained_cfg``.
    - ``onnx`` / ``pytorch``: spatial size from ``classifier.expected_height/width``;
      mean/std identity ([0,1]), except ``pytorch`` with ``pytorch_normalization="sdpcrown"``.

    ``override`` (mapping) may set any of: ``size`` (H,W), ``mean``, ``std``,
    ``interpolation``, ``antialias``; it wins over resolved values.
    """
    if classifier_type == "huggingface":
        if classifier_name is None:
            raise ValueError("classifier_name is required to resolve a HuggingFace preproc spec")
        spec = _resolve_huggingface(classifier_name)
    elif classifier_type == "timm":
        spec = _resolve_timm(classifier)
    elif classifier_type in {"onnx", "pytorch"}:
        target_size = (int(classifier.expected_height), int(classifier.expected_width))
        if classifier_type == "pytorch" and pytorch_normalization == "sdpcrown":
            mean, std = SDPCROWN_MEAN, SDPCROWN_STD
        else:
            mean, std = IDENTITY_MEAN, IDENTITY_STD
        spec = PreprocSpec(target_size=target_size, mean=mean, std=std)
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    ov = _as_dict(override)
    if ov:
        spec = PreprocSpec(
            target_size=tuple(int(v) for v in ov["size"]) if "size" in ov else spec.target_size,
            mean=_triple(ov["mean"]) if "mean" in ov else spec.mean,
            std=_triple(ov["std"]) if "std" in ov else spec.std,
            interpolation=str(ov["interpolation"]) if "interpolation" in ov else spec.interpolation,
            antialias=bool(ov["antialias"]) if "antialias" in ov else spec.antialias,
        )
    return spec
