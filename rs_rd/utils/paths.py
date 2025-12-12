"""
Project paths utility module.

This module provides access to important project directories
through environment variables. It can be imported from anywhere in the project.

Required environment variables:
    - PRJS: Base path for datasets and models
    - RSLTS: Base path for results
"""

import os
from pathlib import Path


def _get_env_var(var_name: str) -> str:
    """
    Get an environment variable or raise an error if not set.

    Args:
        var_name: Name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set.
    """
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"${var_name} environment variable not set")
    return value


def get_dataset_dir(dataset_name: str = "CIFAR-10") -> Path:
    """
    Get the dataset directory.

    Args:
        dataset_name: Name of the dataset. Defaults to "CIFAR-10".

    Returns:
        Path object pointing to the dataset directory.

    Raises:
        ValueError: If $PRJS environment variable is not set.
    """
    prjs = _get_env_var("PRJS")
    return Path(prjs) / "datasets" / dataset_name


def get_models_dir(dataset_name: str = "CIFAR-10") -> Path:
    """
    Get the models directory for a specific dataset.

    Args:
        dataset_name: Name of the dataset. Defaults to "CIFAR-10".

    Returns:
        Path object pointing to the models directory.

    Raises:
        ValueError: If $PRJS environment variable is not set.
    """
    prjs = _get_env_var("PRJS")
    return Path(prjs) / "models" / dataset_name


def get_results_dir(dataset_name: str = "CIFAR-10") -> Path:
    """
    Get the results directory for a specific dataset.

    Args:
        dataset_name: Name of the dataset. Defaults to "CIFAR-10".

    Returns:
        Path object pointing to the results directory.

    Raises:
        ValueError: If $RSLTS environment variable is not set.
    """
    rscode = _get_env_var("RSLTS")
    return Path(rscode) / dataset_name


def get_prjs_base_dir() -> Path:
    """
    Get the base PRJS directory.

    Returns:
        Path object pointing to the base PRJS directory.

    Raises:
        ValueError: If $PRJS environment variable is not set.
    """
    prjs = _get_env_var("PRJS")
    return Path(prjs)


def get_results_base_dir() -> Path:
    """
    Get the base RSLTS directory.

    Returns:
        Path object pointing to the base RSLTS directory.

    Raises:
        ValueError: If $RSLTS environment variable is not set.
    """
    rscode = _get_env_var("RSLTS")
    return Path(rscode)