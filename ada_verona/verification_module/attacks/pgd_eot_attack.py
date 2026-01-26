# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class EOTPGDAttack(Attack):
    """
    PGD (L2 or L_inf) with Expectation over Transformation (EOT).

    This is a minimal extension of `PGDAttack`:
    - Still performs standard PGD projection + optional random start.
    - Replaces the single forward/backward with an EOT-averaged loss:
        loss = mean_k CE(model(x_adv)_k, target), where each forward is stochastic.

    Notes:
    - `bounds=None` means *no clamping* (useful if inputs are already normalized).
    - `std_rescale_factor` converts pixel-space epsilons to model-input-space epsilons, matching `PGDAttack`.
      E.g. if inputs are divided by std=0.225, set std_rescale_factor=0.225.
    """

    Norm = Literal["l2", "inf"]

    def __init__(
        self,
        *,
        number_iterations: int,
        eot_samples: int = 20,
        rel_stepsize: float | None = None,
        abs_stepsize: float | None = None,
        randomise: bool = False,
        norm: Norm = "l2",
        bounds: tuple[float, float] | None = (0.0, 1.0),
        std_rescale_factor: float | None = None,
    ) -> None:
        super().__init__()
        if number_iterations <= 0:
            raise ValueError("number_iterations must be > 0")
        if eot_samples <= 0:
            raise ValueError("eot_samples must be > 0")
        if norm not in {"l2", "inf"}:
            raise ValueError("norm must be one of {'l2', 'inf'}")

        self.number_iterations = number_iterations
        self.eot_samples = eot_samples
        self.randomise = randomise
        self.norm = norm
        self.bounds = bounds
        self.std_rescale_factor = std_rescale_factor

        self.abs_stepsize = abs_stepsize
        self.rel_stepsize = rel_stepsize

        bounds_str = f"bounds={self.bounds}" if self.bounds is not None else "bounds=None"
        rescale_str = f", std_rescale_factor={self.std_rescale_factor}" if self.std_rescale_factor is not None else ""
        self.name = (
            f"EOTPGDAttack (iterations={self.number_iterations}, eot_samples={self.eot_samples}, "
            f"rel_stepsize={self.rel_stepsize}, abs_stepsize={self.abs_stepsize}, "
            f"randomise={self.randomise}, norm={self.norm}, {bounds_str}{rescale_str})"
        )

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        was_training = model.training
        model.eval()

        loss_fn = nn.CrossEntropyLoss()
        adv_images = data.clone().detach()

        # Convert pixel-space epsilon into model-input-space epsilon if inputs are normalized.
        if self.std_rescale_factor is not None:
            epsilon = epsilon / self.std_rescale_factor

        if self.abs_stepsize is not None:
            step_size = self.abs_stepsize
        else:
            rel_stepsize = self.rel_stepsize
            if rel_stepsize is None:
                rel_stepsize = (2.5 if self.norm == "l2" else 1.0) / self.number_iterations
            step_size = rel_stepsize * epsilon

        if self.randomise:
            if self.norm == "l2":
                delta = torch.randn_like(data)
                if delta.dim() == 1:
                    delta_norm = torch.norm(delta, p=2) + 1e-10
                    delta = delta * (epsilon / delta_norm)
                else:
                    delta_flat = delta.view(delta.shape[0], -1)
                    delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-10
                    delta = delta * (epsilon / delta_norm).view(-1, *([1] * (len(delta.shape) - 1)))
                adv_images = data + delta
            else:
                adv_images = adv_images + torch.empty_like(data).uniform_(-epsilon, epsilon)
            if self.bounds is not None:
                adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
            else:
                adv_images = adv_images.detach()

        try:
            for _ in range(self.number_iterations):
                adv_images.requires_grad = True

                # EOT: average loss over multiple stochastic forwards.
                loss = 0.0
                for _ in range(self.eot_samples):
                    output = model(adv_images)
                    loss = loss + loss_fn(output, target)
                loss = loss / float(self.eot_samples)

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
                else:
                    adv_images = adv_images.detach() + step_size * grad.sign()
                    delta = torch.clamp(adv_images - data, min=-epsilon, max=epsilon)
                    adv_images = data + delta

                if self.bounds is not None:
                    adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
                else:
                    adv_images = adv_images.detach()

            return adv_images
        finally:
            if was_training:
                model.train()
