"""
Restarting PGD attack with early stopping.

This attack is intended for the VERONA attack-estimation pipeline:
- It supports multiple random restarts.
- It can stop immediately once an adversarial example is found (SAT).
- It tracks time spent in the iterative inner loop (sum over executed iterations).

Note: early-stop adversariality is checked using top-1 by default (configurable via `top_k`).
"""

from __future__ import annotations

import time

import torch
from torch import Tensor, nn
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class RestartingPGDAttack(Attack):
    """PGD with multiple restarts and early stopping on adversariality."""

    def __init__(
        self,
        *,
        number_iterations: int,
        n_restarts: int = 1,
        rel_stepsize: float | None = None,
        abs_stepsize: float | None = None,
        step_size: float | None = None,  # Deprecated: use abs_stepsize instead
        randomise: bool = False,
        norm: str = "inf",
        bounds: tuple[float, float] | None = None,
        std_rescale_factor: float | None = None,
        top_k: int = 1,
        early_stop_on_success: bool = True,
    ) -> None:
        super().__init__()

        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self.number_iterations = number_iterations
        self.n_restarts = n_restarts
        self.randomise = randomise
        self.norm = norm
        self.bounds = bounds
        self.std_rescale_factor = std_rescale_factor
        self.top_k = top_k
        self.early_stop_on_success = early_stop_on_success

        # backward compatibility: step_size -> abs_stepsize
        if step_size is not None:
            if abs_stepsize is not None:
                raise ValueError("Cannot specify both step_size and abs_stepsize. Use abs_stepsize only.")
            abs_stepsize = step_size

        self.abs_stepsize = abs_stepsize
        self.rel_stepsize = rel_stepsize

        self.last_inner_loop_seconds: float | None = None

        bounds_str = f"bounds={self.bounds}" if self.bounds is not None else "bounds=None"
        rescale_str = f", std_rescale_factor={self.std_rescale_factor}" if self.std_rescale_factor is not None else ""
        self.name = (
            f"RestartingPGDAttack (iterations={self.number_iterations}, restarts={self.n_restarts}, "
            f"rel_stepsize={self.rel_stepsize}, abs_stepsize={self.abs_stepsize}, "
            f"randomise={self.randomise}, norm={self.norm}, {bounds_str}{rescale_str}, "
            f"top_k={self.top_k}, early_stop={self.early_stop_on_success})"
        )

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        # adapted from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
        # l2 norm adapted from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py
        was_training = model.training
        model.eval()

        # Reset timing for this execute call.
        inner_loop_seconds = 0.0

        # Convert pixel-space epsilon into model-input-space epsilon (for normalized models).
        if self.norm in {"l2", "inf"} and self.std_rescale_factor is not None:
            epsilon = epsilon / self.std_rescale_factor

        if self.abs_stepsize is not None:
            step_size = self.abs_stepsize
        else:
            rel_stepsize = self.rel_stepsize
            if rel_stepsize is None:
                rel_stepsize = (2.5 if self.norm == "l2" else 1.0) / self.number_iterations
            step_size = rel_stepsize * epsilon

        loss_fn = nn.CrossEntropyLoss()

        def is_adversarial(x: Tensor) -> bool:
            with torch.no_grad():
                logits = model(x)
                _, predicted_labels = torch.topk(logits, self.top_k)
                # predicted_labels shape: (batch, top_k) or (top_k,) depending on model output
                # target is expected to be shape (batch,)
                if predicted_labels.dim() == 1:
                    return int(target.item()) not in predicted_labels.cpu().tolist()
                return not torch.any(predicted_labels.eq(target.view(-1, 1))).item()

        def init_adv_images() -> Tensor:
            adv = data.clone().detach()
            if not self.randomise:
                return adv

            if self.norm == "l2":
                delta = torch.randn_like(data)
                if delta.dim() == 1:
                    delta_norm = torch.norm(delta, p=2) + 1e-10
                    delta = delta * (epsilon / delta_norm)
                else:
                    delta_flat = delta.view(delta.shape[0], -1)
                    delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-10
                    delta = delta * (epsilon / delta_norm).view(-1, *([1] * (len(delta.shape) - 1)))
                adv = data + delta
            else:
                adv = adv + torch.empty_like(data).uniform_(-epsilon, epsilon)

            if self.bounds is not None:
                adv = torch.clamp(adv, min=self.bounds[0], max=self.bounds[1]).detach()
            else:
                adv = adv.detach()
            return adv

        best_adv = data.clone().detach()
        best_loss = -float("inf")

        try:
            for _restart in range(self.n_restarts):
                adv_images = init_adv_images()

                # If the random start already fools the classifier, stop immediately.
                if self.early_stop_on_success and is_adversarial(adv_images):
                    self.last_inner_loop_seconds = inner_loop_seconds
                    return adv_images

                for _ in range(self.number_iterations):
                    t0 = time.perf_counter()

                    adv_images.requires_grad = True
                    output = model(adv_images)
                    loss = loss_fn(output, target)
                    grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

                    if self.norm == "l2":
                        if grad.dim() == 1:
                            grad_norm = torch.norm(grad, p=2) + 1e-10
                            normalized_grad = grad / grad_norm
                        else:
                            grad_flat = grad.view(grad.shape[0], -1)
                            grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-10
                            normalized_grad = grad / grad_norm.view(-1, *([1] * (len(grad.shape) - 1)))
                        adv_images = adv_images.detach() + step_size * normalized_grad

                        delta = adv_images - data
                        if delta.dim() == 1:
                            delta_norm = torch.norm(delta, p=2) + 1e-10
                            delta = delta * min(1.0, float(epsilon / delta_norm))
                        else:
                            delta_flat = delta.view(delta.shape[0], -1)
                            delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-10
                            delta = delta * torch.min(torch.ones_like(delta_norm), epsilon / delta_norm).view(
                                -1, *([1] * (len(delta.shape) - 1))
                            )
                        adv_images = data + delta
                        if self.bounds is not None:
                            adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
                        else:
                            adv_images = adv_images.detach()
                    else:
                        adv_images = adv_images.detach() + step_size * grad.sign()
                        delta = torch.clamp(adv_images - data, min=-epsilon, max=epsilon)
                        adv_images = data + delta
                        if self.bounds is not None:
                            adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
                        else:
                            adv_images = adv_images.detach()

                    if self.early_stop_on_success and is_adversarial(adv_images):
                        inner_loop_seconds += time.perf_counter() - t0
                        self.last_inner_loop_seconds = inner_loop_seconds
                        return adv_images

                    inner_loop_seconds += time.perf_counter() - t0

                # No success for this restart; keep the strongest (highest CE loss) candidate.
                with torch.no_grad():
                    final_loss = float(loss_fn(model(adv_images), target).item())
                if final_loss > best_loss:
                    best_loss = final_loss
                    best_adv = adv_images

            self.last_inner_loop_seconds = inner_loop_seconds
            return best_adv
        finally:
            self.last_inner_loop_seconds = inner_loop_seconds
            if was_training:
                model.train()
