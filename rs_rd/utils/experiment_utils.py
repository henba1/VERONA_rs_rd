"""
Utility functions for experiment scripts.

This module contains reusable utility functions for setting up and running
VERONA experiments, including dataset sampling, configuration management,
and distribution creation.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torchvision import datasets, transforms

from ada_verona import (
    DatasetSampler,
    EpsilonValueEstimator,
    ExperimentDataset,
    ExperimentRepository,
    Network,
    ONNXNetwork,
    PropertyGenerator,
    PyTorchNetwork,
)


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
        "ImageNet": {"class": datasets.ImageNet, "default_size": (224, 224), "channels": 3, "num_classes": 1000},
    }


def get_balanced_sample(
    dataset_name="CIFAR-10", train_bool=True, dataset_size=100, dataset_dir=None, seed=42, image_size=None, flatten=True
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

    Returns:
        tuple: (balanced_dataset, balanced_sample_idx)
            - balanced_dataset: PyTorch Subset with balanced samples
            - balanced_sample_idx: Indices of selected samples

    Raises:
        ValueError: If dataset_name is not supported
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = get_dataset_config()

    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {', '.join(dataset_config.keys())}")

    config = dataset_config[dataset_name]
    target_size = image_size if image_size is not None else config["default_size"]

    transform_list = [transforms.Resize(target_size), transforms.ToTensor()]
    if flatten:
        transform_list.append(torch.flatten)
    data_transforms = transforms.Compose(transform_list)

    dataset_class = config["class"]

    torch_dataset = dataset_class(root=dataset_dir, train=train_bool, download=False, transform=data_transforms)

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
    dataset_name="CIFAR-10", train_bool=True, dataset_size=100, dataset_dir=None, seed=42, image_size=None, flatten=True
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

    config = dataset_config[dataset_name]
    target_size = image_size if image_size is not None else config["default_size"]

    transform_list = [transforms.Resize(target_size), transforms.ToTensor()]
    if flatten:
        transform_list.append(torch.flatten)
    data_transforms = transforms.Compose(transform_list)

    dataset_class = config["class"]

    torch_dataset = dataset_class(root=dataset_dir, train=train_bool, download=False, transform=data_transforms)

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

    # Load ONNX models
    for model_path in sorted(models_dir.glob("*.onnx")):
        # Check if file exists (handles symlinks correctly)
        if model_path.exists() and (model_path.is_file() or model_path.is_symlink()):
            try:
                network_list.append(ONNXNetwork(path=model_path))
                logging.info(f"Loaded ONNX model: {model_path.name}")
            except Exception as e:
                logging.warning(f"Failed to load ONNX model {model_path.name}: {e}")

    # Load PyTorch models
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
) -> Path:
    """
    Create an experiment directory with standardized naming convention.

    The directory name follows the pattern: experiment_type_dataset_name_timestamp

    Args:
        results_dir: Base results directory where experiment folder will be created
        experiment_type: Type of experiment (e.g., "adv_attack", "certification")
        dataset_name: Name of the dataset (e.g., "CIFAR-10", "MNIST")
        timestamp: Optional timestamp string. If None, will be generated.

    Returns:
        Path to the created experiment directory
    """
    from datetime import datetime

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create safe dataset name (lowercase, dashes replaced with underscores)
    dataset_name_safe = dataset_name.lower().replace("-", "_")

    experiment_dir_name = f"{experiment_type}_{dataset_name_safe}_{timestamp}"
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
