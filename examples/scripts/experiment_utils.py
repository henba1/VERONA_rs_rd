"""
Utility functions for experiment scripts.

This module contains reusable utility functions for setting up and running
VERONA experiments, including dataset sampling, configuration management,
and distribution creation.
"""

import logging

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
    PropertyGenerator,
)


def get_balanced_sample(
    dataset_name="CIFAR-10", train_bool=True, dataset_size=100, dataset_dir=None, seed=42, image_size=None
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

    Returns:
        tuple: (balanced_dataset, balanced_sample_idx)
            - balanced_dataset: PyTorch Subset with balanced samples
            - balanced_sample_idx: Indices of selected samples

    Raises:
        ValueError: If dataset_name is not supported
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dataset configuration mapping
    dataset_config = {
        "CIFAR-10": {"class": datasets.CIFAR10, "default_size": (32, 32), "channels": 3},
        "MNIST": {"class": datasets.MNIST, "default_size": (28, 28), "channels": 1},
        "ImageNet": {"class": datasets.ImageNet, "default_size": (224, 224), "channels": 3},
    }

    if dataset_name not in dataset_config:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. " f"Supported datasets: {', '.join(dataset_config.keys())}"
        )

    config = dataset_config[dataset_name]
    target_size = image_size if image_size is not None else config["default_size"]

    # Create transforms
    data_transforms = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor(), torch.flatten])

    dataset_class = config["class"]

    if train_bool:
        torch_dataset = dataset_class(root=dataset_dir, train=True, download=False, transform=data_transforms)

        # Extract labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # Use StratifiedShuffleSplit to create balanced subsets
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=dataset_size, random_state=seed)
        for train_idx, _ in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = train_idx

    else:
        torch_dataset = dataset_class(root=dataset_dir, train=False, download=False, transform=data_transforms)

        # Extract labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # Use StratifiedShuffleSplit to create balanced subsets
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=dataset_size, random_state=seed)
        for _, test_idx in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = test_idx

    # Create subset of original dataset using the balanced indices
    balanced_dataset = Subset(torch_dataset, balanced_sample_idx)

    return balanced_dataset, balanced_sample_idx


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


def create_distribution(
    experiment_repository: ExperimentRepository,
    dataset: ExperimentDataset,
    dataset_sampler: DatasetSampler,
    epsilon_value_estimator: EpsilonValueEstimator,
    property_generator: PropertyGenerator,
    config=None,
):
    """
    Create robustness distribution for all networks in the experiment repository.

    This function iterates through all networks, samples data points, computes
    epsilon values, and saves results to the experiment repository.

    Args:
        experiment_repository: Repository for managing experiment data and results
        dataset: Dataset to sample from
        dataset_sampler: Sampler for selecting data points
        epsilon_value_estimator: Estimator for computing epsilon values
        property_generator: Generator for creating verification properties
        config: Optional configuration dictionary for network-specific settings
    """
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

    experiment_repository.save_plots()
    logging.info(f"Failed for networks: {failed_networks}")

