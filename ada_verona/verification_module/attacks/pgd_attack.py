import torch
from torch import Tensor, nn
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class PGDAttack(Attack):
    """
    A class to perform the Projected Gradient Descent (PGD) attack.

    Attributes:
        number_iterations (int): The number of iterations for the attack.
        step_size (float): The step size for each iteration.
        randomise (bool): Whether to randomize the initial perturbation.
        norm (str): The norm to use ('inf' or 'l2').
    """

    def __init__(
        self, number_iterations: int, step_size: float = None, randomise: bool = False, norm: str = "inf"
    ) -> None:
        """
        Initialize the PGDAttack with specific parameters.

        Args:
            number_iterations (int): The number of iterations for the attack.
            step_size (float, optional): The step size for each iteration. Defaults to None.
            randomise (bool, optional): Whether to randomize the initial perturbation. Defaults to False.
            norm (str, optional): The norm to use ('inf' or 'l2'). Defaults to 'inf'.
        """
        super().__init__()
        self.number_iterations = number_iterations
        self.step_size = step_size
        self.randomise = randomise
        self.norm = norm
        self.name = (
            f"PGDAttack (iterations={self.number_iterations}, "
            f"step_size={self.step_size}, randomise={self.randomise}, norm={self.norm})"
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

        step_size = self.step_size
        if not step_size:
            if self.norm == "l2":
                step_size = 2.5 * epsilon / self.number_iterations
            else:
                step_size = epsilon / self.number_iterations

        if self.randomise:
            if self.norm == "l2":
                delta = torch.randn_like(data)
                delta_flat = delta.view(delta.shape[0], -1)
                delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
                delta = delta * epsilon / delta_norm.view(-1, *([1] * (len(delta.shape) - 1)))
                adv_images = data + delta
            else:
                adv_images = adv_images + torch.empty_like(data).uniform_(-epsilon, epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

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
                adv_images = torch.clamp(data + delta, min=0, max=1).detach()
            else:
                adv_images = adv_images.detach() + step_size * grad.sign()
                delta = torch.clamp(adv_images - data, min=-epsilon, max=epsilon)
                adv_images = torch.clamp(data + delta, min=0, max=1).detach()

        return adv_images
