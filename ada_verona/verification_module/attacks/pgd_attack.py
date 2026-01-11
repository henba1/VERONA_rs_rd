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

import torch
from torch import Tensor, nn
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class PGDAttack(Attack):
    """
    A class to perform the Projected Gradient Descent (PGD) attack.

    Attributes:
        number_iterations (int): The number of iterations for the attack.
        rel_stepsize (float): Step size relative to epsilon. If None, uses default based on norm.
        abs_stepsize (float, optional): Absolute step size. If given, takes precedence over rel_stepsize.
        randomise (bool): Whether to randomize the initial perturbation.
        norm (str): The norm to use ('inf' or 'l2').
    """

    def __init__(
        self,
        number_iterations: int,
        rel_stepsize: float = None,
        abs_stepsize: float = None,
        step_size: float = None,  # Deprecated: use abs_stepsize instead
        randomise: bool = False,
        norm: str = "inf",
        bounds: tuple = None,
    ) -> None:
        """
        Initialize the PGDAttack with specific parameters.

        Args:
            number_iterations (int): The number of iterations for the attack.
            rel_stepsize (float, optional): Step size relative to epsilon. If None, uses default:
                - For L2: 2.5 / number_iterations (to match original behavior)
                - For Linf: 1.0 / number_iterations
            abs_stepsize (float, optional): Absolute step size. If given, takes precedence over rel_stepsize.
            step_size (float, optional): Deprecated alias for abs_stepsize. Use abs_stepsize instead.
            randomise (bool, optional): Whether to randomize the initial perturbation. Defaults to False.
            norm (str, optional): The norm to use ('inf' or 'l2'). Defaults to 'inf'.
            bounds (tuple, optional): (min, max) bounds for clamping perturbed images.
                If None, no bounds clamping is applied (useful for normalized images).
                Defaults to None.
        """
        super().__init__()
        self.number_iterations = number_iterations
        self.randomise = randomise
        self.norm = norm
        self.bounds = bounds

        # backward compatibility: step_size -> abs_stepsize
        if step_size is not None:
            if abs_stepsize is not None:
                raise ValueError("Cannot specify both step_size and abs_stepsize. Use abs_stepsize only.")
            abs_stepsize = step_size

        self.abs_stepsize = abs_stepsize
        self.rel_stepsize = rel_stepsize

        bounds_str = f"bounds={self.bounds}" if self.bounds is not None else "bounds=None"
        self.name = (
            f"PGDAttack (iterations={self.number_iterations}, "
            f"rel_stepsize={self.rel_stepsize}, abs_stepsize={self.abs_stepsize}, "
            f"randomise={self.randomise}, norm={self.norm}, {bounds_str})"
        )

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the PGD attack on the given model and data.

        Args:
            model (Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude.

        Returns:
            Tensor: The perturbed data.
        """
        # adapted from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
        # l2 norm adapted from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py
        loss_fn = nn.CrossEntropyLoss()
        adv_images = data.clone().detach()

        step_size = self.abs_stepsize if self.abs_stepsize is not None else self.rel_stepsize * epsilon

        if self.randomise:
            if self.norm == "l2":
                delta = torch.randn_like(data)
                delta_flat = delta.view(delta.shape[0], -1)
                delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
                delta = delta * epsilon / delta_norm.view(-1, *([1] * (len(delta.shape) - 1)))
                adv_images = data + delta
            else:
                adv_images = adv_images + torch.empty_like(data).uniform_(-epsilon, epsilon)
            if self.bounds is not None:
                adv_images = torch.clamp(adv_images, min=self.bounds[0], max=self.bounds[1]).detach()
            else:
                adv_images = adv_images.detach()

        for _ in range(self.number_iterations):
            adv_images.requires_grad = True
            output = model(adv_images)

            loss = loss_fn(output, target)
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            if self.norm == "l2":
                grad_flat = grad.view(grad.shape[0], -1)
                grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-10
                normalized_grad = grad / grad_norm.view(-1, *([1] * (len(grad.shape) - 1)))
                adv_images = adv_images.detach() + step_size * normalized_grad

                delta = adv_images - data
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

        return adv_images
