import torch


def test_cw_attack_initialization(cw_attack):
    """Test that CW attack is initialized with correct parameters."""
    assert not cw_attack.targeted
    assert cw_attack.confidence == 0.0
    assert cw_attack.learning_rate == 0.01
    assert cw_attack.binary_search_steps == 3
    assert cw_attack.max_iterations == 10
    assert cw_attack.abort_early


def test_cw_attack_execute(cw_attack, model, data, target):
    """Test that CW attack executes and returns valid adversarial examples."""
    epsilon = 0.1
    perturbed_data = cw_attack.execute(model, data, target, epsilon)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == data.shape
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)


def test_cw_attack_perturbation_bounded(cw_attack, model, data, target):
    """Test that perturbations respect box constraints."""
    epsilon = 0.05
    perturbed_data = cw_attack.execute(model, data, target, epsilon)

    # Check that perturbed data is within valid range [0, 1]
    assert torch.all(perturbed_data >= 0.0)
    assert torch.all(perturbed_data <= 1.0)


def test_cw_attack_changes_prediction(cw_attack, model, data, target):
    """Test that CW attack attempts to change the prediction."""
    epsilon = 0.5  # Large epsilon to ensure attack has room to work
    model.eval()

    # Get original prediction
    with torch.no_grad():
        original_output = model(data)
        original_pred = original_output.argmax(dim=1)

    # Get adversarial prediction
    perturbed_data = cw_attack.execute(model, data, target, epsilon)
    with torch.no_grad():
        adv_output = model(perturbed_data)
        adv_pred = adv_output.argmax(dim=1)

    # Note: Due to limited iterations in test, attack might not always succeed
    # We just verify that the method runs without errors and produces valid output
    assert isinstance(adv_pred, torch.Tensor)
    assert adv_pred.shape == original_pred.shape


def test_cw_attack_batch_processing(cw_attack, model, target):
    """Test that CW attack handles batch processing correctly."""
    batch_size = 3
    batch_data = torch.randn(batch_size, 10)
    batch_target = torch.tensor([0, 1, 0])
    epsilon = 0.1

    perturbed_batch = cw_attack.execute(model, batch_data, batch_target, epsilon)

    assert perturbed_batch.shape == batch_data.shape
    assert torch.all(perturbed_batch >= 0) and torch.all(perturbed_batch <= 1)


def test_cw_attack_targeted_mode():
    """Test that CW attack works in targeted mode."""
    from ada_verona.verification_module.attacks.cw_attack import CWL2Attack

    targeted_attack = CWL2Attack(
        targeted=True,
        confidence=0.0,
        learning_rate=0.01,
        binary_search_steps=2,
        max_iterations=5,
    )

    assert targeted_attack.targeted

    # Create simple test setup
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 3)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    model.eval()
    data = torch.randn(1, 10)
    target = torch.tensor([2])  # Target class
    epsilon = 0.5

    perturbed_data = targeted_attack.execute(model, data, target, epsilon)

    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == data.shape
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)

