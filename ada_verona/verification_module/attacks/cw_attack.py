import torch
from torch import Tensor
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class CWL2Attack(Attack):
    """
    A class to perform the Carlini & Wagner L2 attack.

    The C&W L2 attack is an optimization-based adversarial attack that minimizes the L2 distance
    between the original and adversarial examples while ensuring misclassification. It uses a
    change of variables (tanh) to handle box constraints and binary search over a constant c
    that balances the distance and misclassification objectives.

    Reference:
        Nicholas Carlini and David Wagner. "Towards Evaluating the Robustness of Neural Networks."
        IEEE Symposium on Security and Privacy (S&P), 2017.
        https://arxiv.org/abs/1608.04644

    Attributes:
        targeted (bool): Whether to perform a targeted attack.
        confidence (float): Confidence parameter (kappa) for the attack objective.
        learning_rate (float): Learning rate for the Adam optimizer.
        binary_search_steps (int): Number of binary search steps for the constant c.
        max_iterations (int): Maximum number of optimization iterations per binary search step.
        abort_early (bool): Whether to abort early if attack succeeds.
        initial_const (float): Initial value for the constant c.
    """

    def __init__(
        self,
        targeted: bool = False,
        confidence: float = 0.0,
        learning_rate: float = 0.01,
        binary_search_steps: int = 9,
        max_iterations: int = 1000,
        abort_early: bool = True,
        initial_const: float = 1e-3,
    ) -> None:
        """
        Initialize the CWL2Attack with specific parameters.

        Args:
            targeted (bool, optional): Whether to perform a targeted attack. Defaults to False.
            confidence (float, optional): Confidence parameter (kappa) for the attack. Defaults to 0.0.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            binary_search_steps (int, optional): Number of binary search steps. Defaults to 9.
            max_iterations (int, optional): Maximum optimization iterations. Defaults to 1000.
            abort_early (bool, optional): Whether to abort early on success. Defaults to True.
            initial_const (float, optional): Initial value for constant c. Defaults to 1e-3.
        """
        super().__init__()
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.name = (
            f"CWL2Attack (targeted={self.targeted}, confidence={self.confidence}, "
            f"lr={self.learning_rate}, binary_search_steps={self.binary_search_steps}, "
            f"max_iterations={self.max_iterations})"
        )

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the C&W L2 attack on the given model and data.

        Args:
            model (Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude (used as constraint on L2 distance).

        Returns:
            Tensor: The perturbed data.
        """
        model.eval()
        device = data.device
        batch_size = data.shape[0]

        # Convert target to appropriate shape if needed
        if target.dim() == 0:
            target = target.unsqueeze(0)

        # Initialize variables for tracking best adversarial examples
        best_adv_images = data.clone()
        best_l2_dist = torch.full((batch_size,), float("inf"), device=device)

        # Binary search bounds for the constant c
        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.full((batch_size,), 1e10, device=device)
        const = torch.full((batch_size,), self.initial_const, device=device)

        # Transform data to tanh space for unconstrained optimization
        # x = 0.5 * (tanh(w) + 1) ensures x in [0, 1]
        w = self._inverse_tanh(data)
        w = w.detach().clone().requires_grad_(True)

        # Binary search for optimal constant c
        for _binary_step in range(self.binary_search_steps):
            # Optimizer for this binary search iteration
            optimizer = torch.optim.Adam([w], lr=self.learning_rate)

            # Track best results for this binary search step
            step_best_l2 = torch.full((batch_size,), float("inf"), device=device)
            step_best_adv = data.clone()

            for iteration in range(self.max_iterations):
                optimizer.zero_grad()

                # Transform from w to adversarial image
                adv_images = 0.5 * (torch.tanh(w) + 1)

                # Calculate L2 distance
                l2_dist = torch.sum((adv_images - data).view(batch_size, -1) ** 2, dim=1)

                # Get model predictions
                outputs = model(adv_images)

                # Calculate the attack loss
                loss_f = self._f_loss(outputs, target, self.confidence, self.targeted)

                # Total loss: weighted sum of distance and misclassification
                loss = torch.sum(l2_dist + const * loss_f)

                loss.backward()
                optimizer.step()

                # Update best adversarial examples
                with torch.no_grad():
                    # Check which examples are successful adversarial examples
                    pred_labels = outputs.argmax(dim=1)
                    successful = pred_labels == target if self.targeted else pred_labels != target

                    # Update step best results
                    improved = (l2_dist < step_best_l2) & successful
                    step_best_l2 = torch.where(improved, l2_dist, step_best_l2)
                    for i in range(batch_size):
                        if improved[i]:
                            step_best_adv[i] = adv_images[i].clone()

                    # Update global best results
                    global_improved = (l2_dist < best_l2_dist) & successful
                    best_l2_dist = torch.where(global_improved, l2_dist, best_l2_dist)
                    for i in range(batch_size):
                        if global_improved[i]:
                            best_adv_images[i] = adv_images[i].clone()

                # Early stopping if attack is successful and abort_early is True
                if self.abort_early and iteration % 10 == 0 and successful.all():
                    break

            # Update binary search bounds
            with torch.no_grad():
                for i in range(batch_size):
                    if step_best_l2[i] < float("inf"):
                        # Attack succeeded, decrease const
                        upper_bound[i] = const[i]
                    else:
                        # Attack failed, increase const
                        lower_bound[i] = const[i]

                    # Update const for next binary search iteration
                    if upper_bound[i] < 1e10:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        const[i] = const[i] * 10

        return best_adv_images.detach()

    def _f_loss(self, outputs: Tensor, labels: Tensor, confidence: float, targeted: bool) -> Tensor:
        """
        Calculate the f loss for the C&W attack.

        The loss encourages misclassification (or targeted classification) with a confidence margin.

        Args:
            outputs (Tensor): Model output logits.
            labels (Tensor): Target labels.
            confidence (float): Confidence parameter (kappa).
            targeted (bool): Whether the attack is targeted.

        Returns:
            Tensor: The f loss value.
        """
        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]

        # Get the logit for the target class
        target_logits = outputs.gather(1, labels.view(-1, 1)).squeeze(1)

        # Get the maximum logit among all other classes
        # Create a one-hot mask for the target class
        one_hot_labels = torch.zeros(batch_size, num_classes, device=outputs.device)
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)

        # Mask out the target class and get the max of the rest
        other_logits = outputs - one_hot_labels * 1e10
        other_max_logits = other_logits.max(dim=1)[0]

        if targeted:
            # For targeted attack: maximize target logit relative to others
            # We want: target_logit > other_max_logit + confidence
            # So we minimize: max(0, other_max_logit - target_logit + confidence)
            loss = torch.clamp(other_max_logits - target_logits + confidence, min=0)
        else:
            # For untargeted attack: maximize other logits relative to target
            # We want: other_max_logit > target_logit + confidence
            # So we minimize: max(0, target_logit - other_max_logit + confidence)
            loss = torch.clamp(target_logits - other_max_logits + confidence, min=0)

        return loss

    def _inverse_tanh(self, x: Tensor) -> Tensor:
        """
        Calculate the inverse hyperbolic tangent (arctanh) of x.

        This transforms x from [0, 1] to unconstrained w space.
        x = 0.5 * (tanh(w) + 1) => w = arctanh(2x - 1)

        Args:
            x (Tensor): Input tensor with values in [0, 1].

        Returns:
            Tensor: The arctanh transformed tensor.
        """
        # Clamp to avoid numerical issues at boundaries
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        return torch.atanh(2 * x - 1)
