"""
Utility functions for experiment scripts.

This module contains reusable utility functions for setting up and running
VERONA RS-RD experiments, including dataset sampling, configuration management,
and distribution creation.
"""

import importlib
import importlib.util
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.database.machine_learning_model.network import Network
from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.database.machine_learning_model.pytorch_network import PyTorchNetwork
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator

SDPCROWN_INPUT_STD = 0.225


class SDPCrownCIFAR10Preprocess:
    """
    Preprocessing transform for CIFAR-10 images used by SDP-CROWN.

    Preprocessing used by the SDP paper:
    - Normalizes images using SDP-CROWN specific means and standard deviations
    - Expects input images in [0, 1] range (from ToTensor)
    - Outputs normalized images ready for SDP-CROWN models
    """

    def __init__(self, inception_preprocess=False):
        """
        Args:
            inception_preprocess: If True, uses inception-style preprocessing (2x - 1 scaling).
                                 If False, uses standard SDP-CROWN preprocessing.
        """
        self.inception_preprocess = inception_preprocess
        # SDP-CROWN normalization parameters
        self.means = np.array([125.3, 123.0, 113.9], dtype=np.float32) / 255
        self.std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

        if inception_preprocess:
            # Use 2x - 1 to get [-1, 1]-scaled images
            self.rescaled_means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.rescaled_devs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            self.rescaled_means = self.means
            self.rescaled_devs = self.std

    def __call__(self, image):
        """
        Apply preprocessing to a single image tensor.

        Args:
            image: Tensor image in [0, 1] range with shape (C, H, W)

        Returns:
            Preprocessed image tensor
        """

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        # Convert means and stds to tensors with proper shape for broadcasting
        # Shape: (C,) -> (C, 1, 1) for broadcasting with (C, H, W) tensor
        means_tensor = torch.tensor(self.rescaled_means, dtype=torch.float32).view(3, 1, 1)
        stds_tensor = torch.tensor(self.rescaled_devs, dtype=torch.float32).view(3, 1, 1)

        # Apply normalization: (image - mean) / std
        # Image is in [0, 1] range from ToTensor
        normalized = (image - means_tensor) / stds_tensor

        return normalized


def apply_pytorch_normalization(imgs: torch.Tensor, normalization_type: str) -> torch.Tensor:
    """
    Apply PyTorch normalization to images based on normalization type.

    Args:
        imgs: Image tensor with shape (batch, channels, height, width) or (channels, height, width)
              Expected to be in [0, 1] range
        normalization_type: Type of normalization to apply. Options: "none", "sdpcrown"

    Returns:
        Normalized image tensor

    Raises:
        ValueError: If normalization_type is not supported
    """
    if normalization_type == "sdpcrown":
        means = torch.tensor([125.3, 123.0, 113.9], device=imgs.device, dtype=imgs.dtype) / 255
        stds = torch.tensor([0.225, 0.225, 0.225], device=imgs.device, dtype=imgs.dtype)
        # Handle both batched and unbatched tensors
        if imgs.dim() == 4:  # (batch, channels, height, width)
            return (imgs - means.view(1, 3, 1, 1)) / stds.view(1, 3, 1, 1)
        elif imgs.dim() == 3:  # (channels, height, width)
            return (imgs - means.view(3, 1, 1)) / stds.view(3, 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {imgs.dim()}D tensor")
    elif normalization_type == "none":
        return imgs
    else:
        raise ValueError(f"Unsupported normalization_type: {normalization_type}. Must be one of {{'none', 'sdpcrown'}}")


def get_dataset_config():
    """
    Get dataset configuration mapping.

    Returns:
        dict: Dictionary mapping dataset names to their configuration including:
            - class: PyTorch dataset class
            - default_size: Default image dimensions (width, height)
            - channels: Number of color channels
    """
    return {
        "CIFAR-10": {"class": datasets.CIFAR10, "default_size": (32, 32), "channels": 3, "num_classes": 10},
        "MNIST": {"class": datasets.MNIST, "default_size": (28, 28), "channels": 1, "num_classes": 10},
        "ImageNet": {"class": ImageFolder, "default_size": (224, 224), "channels": 3, "num_classes": 1000},
    }


def build_torchvision_transforms(
    *,
    dataset_name: str,
    image_size: tuple[int, int] | None = None,
    flatten: bool = True,
    sdpcrown_preprocess: bool = False,
) -> transforms.Compose:
    """Create torchvision transforms consistent with existing sampling utilities.

    Args:
        dataset_name: Name of the dataset. Options: "CIFAR-10", "MNIST", "ImageNet"
        image_size: Optional override for (width, height). If None, uses dataset defaults.
        flatten: If True, flatten images to 1D tensors. If False, keep images as (C, H, W).
        sdpcrown_preprocess: If True, apply SDP-CROWN CIFAR-10 preprocessing (only for CIFAR-10).
    """
    dataset_config = get_dataset_config()
    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {', '.join(dataset_config.keys())}")

    config = dataset_config[dataset_name]
    target_size = image_size if image_size is not None else config["default_size"]

    transform_list: list[object] = [transforms.Resize(target_size), transforms.ToTensor()]

    if sdpcrown_preprocess and dataset_name == "CIFAR-10":
        transform_list.append(SDPCrownCIFAR10Preprocess(inception_preprocess=False))

    if flatten:
        transform_list.append(torch.flatten)

    return transforms.Compose(transform_list)


def get_torchvision_dataset(
    *,
    dataset_name: str,
    dataset_dir: str | Path | None,
    train_bool: bool,
    image_size: tuple[int, int] | None = None,
    flatten: bool = True,
    sdpcrown_preprocess: bool = False,
):
    """Return a torchvision dataset with standard preprocessing applied."""
    dataset_config = get_dataset_config()
    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {', '.join(dataset_config.keys())}")

    if dataset_dir is None:
        raise ValueError("dataset_dir must be provided (it cannot be None).")

    data_transforms = build_torchvision_transforms(
        dataset_name=dataset_name,
        image_size=image_size,
        flatten=flatten,
        sdpcrown_preprocess=sdpcrown_preprocess,
    )

    dataset_dir = Path(dataset_dir)
    dataset_class = dataset_config[dataset_name]["class"]

    if dataset_name == "ImageNet":
        split_dir = "train" if train_bool else "val"
        dataset_path = dataset_dir / split_dir
        if not dataset_path.exists():
            raise FileNotFoundError(f"ImageNet {split_dir} directory not found at {dataset_path}")
        return dataset_class(root=str(dataset_path), transform=data_transforms)

    return dataset_class(root=str(dataset_dir), train=train_bool, download=False, transform=data_transforms)


def get_balanced_sample(
    dataset_name="CIFAR-10",
    train_bool=True,
    dataset_size=100,
    dataset_dir=None,
    seed=42,
    image_size=None,
    flatten=True,
    sdpcrown_preprocess=False,
):
    """
    Get a balanced sample from a PyTorch dataset.

    Supports CIFAR-10, MNIST, and ImageNet datasets with stratified sampling
    to ensure balanced class distribution.

    Args:
        dataset_name: Name of the dataset. Options: "CIFAR-10", "MNIST", "ImageNet"
        train_bool: If True, sample from training set; if False, from test set
        dataset_size: Number of samples to select
        dataset_dir: Directory where dataset is stored
        seed: Random seed for reproducibility
        image_size: Target image size (width, height). If None, uses dataset defaults:
                   MNIST: (28, 28), CIFAR-10: (32, 32), ImageNet: (224, 224)
        flatten: If True, flatten images to 1D tensors. If False, keep images as (C, H, W).
                 Default True for backwards compatibility.
        sdpcrown_preprocess: If True, applies SDP-CROWN specific preprocessing for CIFAR-10.
                            Only applies to CIFAR-10 dataset. Default False.

    Returns:
        tuple: (balanced_dataset, balanced_sample_idx)
            - balanced_dataset: PyTorch Subset with balanced samples
            - balanced_sample_idx: Indices of selected samples

    Raises:
        ValueError: If dataset_name is not supported
    """
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
    except ImportError as e:
        raise ImportError("get_balanced_sample requires scikit-learn.") from e

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = get_dataset_config()

    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {', '.join(dataset_config.keys())}")

    torch_dataset = get_torchvision_dataset(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        train_bool=train_bool,
        image_size=image_size,
        flatten=flatten,
        sdpcrown_preprocess=sdpcrown_preprocess,
    )

    # Extract labels
    labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

    # Use StratifiedShuffleSplit with appropriate size parameter based on train_bool
    if train_bool:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=dataset_size, random_state=seed)
        for train_idx, _ in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = train_idx
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=dataset_size, random_state=seed)
        for _, test_idx in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = test_idx

    # Create subset of original dataset using the balanced indices
    balanced_dataset = Subset(torch_dataset, balanced_sample_idx)

    return balanced_dataset, balanced_sample_idx


def get_sample(
    dataset_name="CIFAR-10",
    train_bool=True,
    dataset_size=100,
    dataset_dir=None,
    seed=42,
    image_size=None,
    flatten=True,
    sdpcrown_preprocess=False,
):
    """
    Get a random sample from a PyTorch dataset without stratification.

    Supports CIFAR-10, MNIST, and ImageNet datasets with simple random sampling.
    Unlike get_balanced_sample(), this function does not ensure balanced class distribution.

    Args:
        dataset_name: Name of the dataset. Options: "CIFAR-10", "MNIST", "ImageNet"
        train_bool: If True, sample from training set; if False, from test set
        dataset_size: Number of samples to select
        dataset_dir: Directory where dataset is stored
        seed: Random seed for reproducibility
        image_size: Target image size (width, height). If None, uses dataset defaults:
                   MNIST: (28, 28), CIFAR-10: (32, 32), ImageNet: (224, 224)
        flatten: If True, flatten images to 1D tensors. If False, keep images as (C, H, W).
                 Default True for backwards compatibility.
        sdpcrown_preprocess: If True, applies SDP-CROWN specific preprocessing for CIFAR-10.
                            Only applies to CIFAR-10 dataset. Default False.

    Returns:
        tuple: (sampled_dataset, sample_idx)
            - sampled_dataset: PyTorch Subset with randomly sampled samples
            - sample_idx: Indices of selected samples

    Raises:
        ValueError: If dataset_name is not supported
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = get_dataset_config()

    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {', '.join(dataset_config.keys())}")

    torch_dataset = get_torchvision_dataset(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        train_bool=train_bool,
        image_size=image_size,
        flatten=flatten,
        sdpcrown_preprocess=sdpcrown_preprocess,
    )

    dataset_length = len(torch_dataset)
    if dataset_size > dataset_length:
        raise ValueError(f"Requested sample size {dataset_size} exceeds dataset size {dataset_length}")

    all_indices = np.arange(dataset_length)
    sample_idx = np.random.choice(all_indices, size=dataset_size, replace=False)

    sampled_dataset = Subset(torch_dataset, sample_idx)

    return sampled_dataset, sample_idx


def load_networks_from_directory(models_dir: Path, input_shape: tuple[int], device: torch.device) -> list[Network]:
    """
    Load networks from a directory supporting both ONNX and PyTorch (.pth) models.

    Args:
        models_dir: Directory containing model files
        input_shape: Input shape tuple for PyTorch models (e.g., (1, 3, 32, 32))
        device: PyTorch device to load models on

    Returns:
        List of Network objects (ONNXNetwork or PyTorchNetwork)
    """
    network_list = []

    for model_path in sorted(models_dir.glob("*.onnx")):
        if model_path.exists() and (model_path.is_file() or model_path.is_symlink()):
            try:
                network_list.append(ONNXNetwork(path=model_path))
                logging.info(f"Loaded ONNX model: {model_path.name}")
            except Exception as e:
                logging.warning(f"Failed to load ONNX model {model_path.name}: {e}")

    for model_path in sorted(models_dir.glob("*.pth")):
        if model_path.exists() and (model_path.is_file() or model_path.is_symlink()):
            try:
                loaded = torch.load(model_path, map_location=device, weights_only=False)
                model = loaded.to(device)
                model.eval()

                network_list.append(
                    PyTorchNetwork(
                        model=model,
                        input_shape=input_shape,
                        name=model_path.stem,
                        path=model_path,
                    )
                )
                logging.info(f"Loaded PyTorch model: {model_path.name}")
            except Exception as e:
                logging.warning(f"Failed to load PyTorch model {model_path.name}: {e}")

    return network_list


def get_sdpcrown_models_module():
    """
    Get the SDP-CROWN models module, loading it if necessary.

    Returns:
        The SDP-CROWN models module containing architecture classes.

    Raises:
        ImportError: If the SDP-CROWN models module cannot be loaded.
    """
    try:
        from autoverify.verifier import SDPCrown

        tool_dir = SDPCrown().tool_path
    except Exception:
        tool_dir = Path.home() / ".local/share/autoverify/verifiers/sdpcrown/tool"

    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))

    models_py = tool_dir / "models.py"
    spec = importlib.util.spec_from_file_location("sdpcrown_models", models_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SDP-CROWN models from {models_py}")
    sdpcrown_models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdpcrown_models_module)

    return sdpcrown_models_module


def infer_sdpcrown_architecture(model_name: str, sdpcrown_models_module=None) -> nn.Module:
    """
    Infer PyTorch model architecture from model filename.

    Args:
        model_name: Model filename (without extension)
        sdpcrown_models_module: Optional SDP-CROWN models module. If None, will be loaded automatically.

    Returns:
        Instantiated model architecture

    Raises:
        ImportError: If SDP-CROWN models module cannot be loaded.
        ValueError: If architecture cannot be inferred from model name.
    """
    if sdpcrown_models_module is None:
        sdpcrown_models_module = get_sdpcrown_models_module()

    name = model_name.lower()

    if "mnist" in name:
        if "mlp" in name:
            return sdpcrown_models_module.MNIST_MLP()
        if "convsmall" in name:
            return sdpcrown_models_module.MNIST_ConvSmall()
        return sdpcrown_models_module.MNIST_ConvLarge()

    if "cifar" in name or "cifar10" in name:
        if "cnn_a" in name:
            return sdpcrown_models_module.CIFAR10_CNN_A()
        if "cnn_b" in name:
            return sdpcrown_models_module.CIFAR10_CNN_B()
        if "cnn_c" in name:
            return sdpcrown_models_module.CIFAR10_CNN_C()
        if "convsmall" in name:
            return sdpcrown_models_module.CIFAR10_ConvSmall()
        if "convdeep" in name:
            return sdpcrown_models_module.CIFAR10_ConvDeep()
        if "convlarge" in name or "conv_large" in name:
            return sdpcrown_models_module.CIFAR10_ConvLarge()
        # Default to ConvLarge for CIFAR-10
        return sdpcrown_models_module.CIFAR10_ConvLarge()

    raise ValueError(
        f"Could not infer architecture from model name '{model_name}'. "
        "Please use one of the known SDP-CROWN architectures."
    )


def load_sdpcrown_pytorch_model(model_path: Path, device: torch.device, sdpcrown_models_module=None) -> nn.Module:
    """
    Load a PyTorch model from a .pth file, handling both state_dict and full model objects.

    Args:
        model_path: Path to the .pth file
        device: PyTorch device to load the model on
        sdpcrown_models_module: Optional SDP-CROWN models module. If None, will be loaded automatically
                                 when needed (only for state_dict files).

    Returns:
        Loaded PyTorch model

    Raises:
        ImportError: If SDP-CROWN models module cannot be loaded (when needed for state_dict).
        ValueError: If checkpoint format is unsupported.
    """
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded, (OrderedDict, dict)) and not isinstance(loaded, nn.Module):
        # state_dict, need to instantiate model architecture first
        logging.info(f"Detected state_dict in {model_path.name}, inferring architecture from filename")
        if sdpcrown_models_module is None:
            sdpcrown_models_module = get_sdpcrown_models_module()
        model = infer_sdpcrown_architecture(model_path.stem, sdpcrown_models_module)
        state_dict = OrderedDict(loaded)

        # Some checkpoints are saved from models wrapped with `torch.nn.utils.spectral_norm`,
        # which stores parameters as `*.weight_orig` plus buffers `*.weight_u` and `*.weight_v`.
        # The SDP-CROWN reference architectures expect plain `*.weight`.
        if any(k.endswith(".weight_orig") for k in state_dict):
            logging.info(
                "Detected SpectralNorm-style state_dict keys in %s; remapping `*.weight_orig -> *.weight`.",
                model_path.name,
            )
            remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
            for key, value in state_dict.items():
                if key.endswith(".weight_u") or key.endswith(".weight_v"):
                    continue
                if key.endswith(".weight_orig"):
                    remapped[key[: -len("_orig")]] = value
                    continue
                remapped[key] = value
            state_dict = remapped

        model.load_state_dict(state_dict, strict=True)
    elif isinstance(loaded, nn.Module):
        model = loaded
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {model_path}. "
            "Expected either a state_dict (OrderedDict/dict) or a full model (nn.Module)."
        )

    model = model.to(device)
    model.eval()
    return model


def sdp_crown_models_loading(models_dir: Path, input_shape: tuple[int], device: torch.device) -> list[Network]:
    """
    Load SDP-CROWN PyTorch models from a directory.

    Args:
        models_dir: Directory containing .pth model files
        input_shape: Input shape tuple for PyTorch models (e.g., (1, 3, 32, 32))
        device: PyTorch device to load models on

    Returns:
        List of PyTorchNetwork objects
    """
    sdpcrown_models_module = get_sdpcrown_models_module()

    network_list: list[Network] = []
    for model_path in sorted(models_dir.glob("*.pth")):
        if not (model_path.exists() and (model_path.is_file() or model_path.is_symlink())):
            continue
        try:
            model = load_sdpcrown_pytorch_model(model_path, device, sdpcrown_models_module)
            network_list.append(
                PyTorchNetwork(
                    model=model,
                    input_shape=input_shape,
                    name=model_path.stem,
                    path=model_path,
                )
            )
            logging.info(f"Loaded SDP-CROWN PyTorch model: {model_path.name}")
        except Exception as e:
            logging.warning(f"Failed to load SDP-CROWN PyTorch model {model_path.name}: {e}")

    return network_list


def find_config(network_identifier, config_dict):
    """
    Find configuration path for a given network identifier.

    Args:
        network_identifier: Network identifier object with a path attribute
        config_dict: Dictionary mapping network path patterns to config paths

    Returns:
        Path to configuration file if found, None otherwise
    """
    for key, path in config_dict.items():
        if key in str(network_identifier.path):
            return path
    return None


def create_experiment_directory(
    results_dir: Path | str,
    experiment_type: str,
    dataset_name: str,
    timestamp: str | None = None,
    classifier_name: str | None = None,
    experiment_tag: str | None = None,
) -> Path:
    """
    Create an experiment directory with standardized naming convention.

    The directory name follows the pattern:
    - If experiment_tag is provided: experiment_tag_experiment_type_dataset_name_classifier_name_timestamp
    - If classifier_name is provided: experiment_type_dataset_name_classifier_name_timestamp
    - Otherwise: experiment_type_dataset_name_timestamp

    Args:
        results_dir: Base results directory where experiment folder will be created
        experiment_type: Type of experiment (e.g., "adv_attack", "certification")
        dataset_name: Name of the dataset (e.g., "CIFAR-10", "MNIST")
        timestamp: Optional timestamp string. If None, will be generated.
        classifier_name: Optional classifier name to include in directory name for distinction.
        experiment_tag: Optional experiment tag to prepend to directory name for identification.

    Returns:
        Path to the created experiment directory
    """
    from datetime import datetime

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name_safe = dataset_name.lower().replace("-", "_")

    name_parts = []
    if experiment_tag is not None:
        experiment_tag_safe = experiment_tag.lower().replace("-", "_").replace(" ", "_")
        name_parts.append(experiment_tag_safe)
    name_parts.extend([experiment_type, dataset_name_safe])
    if classifier_name is not None:
        classifier_name_safe = classifier_name.lower().replace("/", "_").replace("-", "_")
        name_parts.append(classifier_name_safe)
    name_parts.append(timestamp)

    experiment_dir_name = "_".join(name_parts)
    experiment_dir_path = results_dir / experiment_dir_name
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    return experiment_dir_path


def save_original_indices(
    dataset_name: str,
    original_indices: np.ndarray,
    output_dir: Path | str,
    sample_size: int,
    split: str | None = None,
) -> Path:
    """
    Save original dataset indices to a text file.

    Creates a file containing the original dataset indices for a sampled subset,
    with appropriate naming and header information.

    Args:
        dataset_name: Name of the dataset (e.g., "CIFAR-10", "MNIST")
        original_indices: Array of original dataset indices
        output_dir: Directory where the indices file should be saved
        sample_size: Number of samples in the subset
        timestamp: Timestamp string for file naming
        split: Optional split identifier (e.g., "train", "test"). If provided,
               included in filename and header
        set_type: Type of dataset set (default: "test"). Used in header message.

    Returns:
        Path to the saved indices file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name_safe = dataset_name.lower().replace("-", "_")

    if split:
        filename = f"{dataset_name_safe}_indices_{split}_nsample_{sample_size}.txt"
        header = f"{dataset_name} {split} indices (n_sample={sample_size})"
    else:
        filename = f"{dataset_name_safe}_indices_nsample_{sample_size}.txt"
        header = f"{dataset_name} indices (n_sample={sample_size})"

    indices_file = output_dir / filename

    np.savetxt(indices_file, original_indices, fmt="%d", header=header)

    if not indices_file.exists():
        raise OSError(f"Failed to write indices file: {indices_file}")

    logging.info(f"Saved {dataset_name} indices to {indices_file}")

    return indices_file


def add_original_indices_to_result_df(results_path, original_indices):
    """
    Add original dataset indices column to result_df.csv.

    Creates a new CSV file with an additional 'original_dataset_index' column
    that maps from image_id (subset index) to the original dataset index.

    Args:
        results_path: Path to the results directory containing result_df.csv
        original_indices: Array/list mapping subset index to original dataset index

    Returns:
        Path to the new result file, or None if result_df.csv doesn't exist
    """
    import pandas as pd

    result_csv_path = results_path / "result_df.csv"

    if not result_csv_path.exists():
        logging.warning(f"result_df.csv not found at {result_csv_path}")
        return None

    # Read the result dataframe
    df = pd.read_csv(result_csv_path)
    logging.info(f"Loaded result_df with {len(df)} rows")

    # Create mapping from subset index (image_id) to original index
    idx_mapping = {i: int(original_indices[i]) for i in range(len(original_indices))}

    # Add original_dataset_index column
    if "image_id" in df.columns:
        df["original_dataset_index"] = df["image_id"].map(idx_mapping)
        logging.info("Added 'original_dataset_index' column to result_df")
    else:
        logging.warning("Column 'image_id' not found in result_df")
        return None

    # Save with new name
    output_path = results_path / "result_df_with_original_idx.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Saved result_df with original indices to: {output_path}")

    return output_path


def rescale_eps_in_results(*, results_path: Path, std: float = SDPCROWN_INPUT_STD) -> None:
    """Rewrite result CSVs so stored eps values are in model-input space.

    When inputs are normalized by dividing by `std` (e.g. CIFAR-10 SDP-CROWN preprocessing),
    epsilons in *pixel space* must be converted to *model-input space* as:

    \[
        \epsilon_\text{model} = \epsilon_\text{pixel} / \text{std}.
    \]

    The rescaled columns overwrite the existing ones, while keeping the originally stored
    values in `*_raw` columns for traceability.
    """
    import pandas as pd

    def _rescale_csv(csv_path: Path, *, epsilon_cols: list[tuple[str, str]]) -> None:
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path, index_col=0)
        changed = False

        for col, raw_col in epsilon_cols:
            if col not in df.columns:
                continue
            if raw_col not in df.columns:
                df[raw_col] = df[col]
                changed = True

            # Idempotent: always recompute from raw.
            df[col] = df[raw_col] / std
            changed = True

        if changed:
            df.to_csv(csv_path)

    _rescale_csv(
        results_path / "result_df.csv",
        epsilon_cols=[
            ("epsilon_value", "epsilon_value_raw"),
            ("smallest_sat_value", "smallest_sat_value_raw"),
        ],
    )
    _rescale_csv(
        results_path / "per_epsilon_results.csv",
        epsilon_cols=[
            ("epsilon_value", "epsilon_value_raw"),
        ],
    )


def create_distribution(
    experiment_repository: ExperimentRepository,
    dataset: ExperimentDataset,
    dataset_sampler: DatasetSampler,
    epsilon_value_estimator: EpsilonValueEstimator,
    property_generator: PropertyGenerator,
    config=None,
    network_list=None,
):
    """
    Create robustness distribution for all networks in the experiment repository.

    This function iterates through all networks, samples data points, computes
    epsilon values, and saves results to the experiment repository.w

    Args:
        experiment_repository: Repository for managing experiment data and results
        dataset: Dataset to sample from
        dataset_sampler: Sampler for selecting data points
        epsilon_value_estimator: Estimator for computing epsilon values
        property_generator: Generator for creating verification properties
        config: Optional configuration dictionary for network-specific settings
        network_list: Optional explicit list of networks to verify. If None,
            networks are loaded from the experiment repository.
    """
    if network_list is None:
        network_list = experiment_repository.get_network_list()
    failed_networks = []
    for network in network_list:
        if config is not None:
            epsilon_value_estimator.verifier.config = find_config(network, config)
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
        except Exception as e:
            logging.info(f"failed for network: {network} with error: {e}")
            failed_networks.append(network)
            continue
        for data_point in sampled_data:
            verification_context = experiment_repository.create_verification_context(
                network, data_point, property_generator
            )

            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

            experiment_repository.save_result(epsilon_value_result)

    # Try to save plots, cont upon error (e.g., due to insufficient data for KDE)
    try:
        experiment_repository.save_plots()
    except Exception as e:
        logging.warning(f"Failed to save plots: {e}")
        logging.info("Continuing without plots - results CSV files are still available")

    logging.info(f"Failed for networks: {failed_networks}")
