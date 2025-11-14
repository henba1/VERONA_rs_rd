"""
Comet ML tracking utilities for VERONA experiments.

This module provides reusable functions for tracking experiments with Comet ML,
including initialization, logging parameters, metrics, and assets.

Example usage:
    ```python
    from comet_tracker import CometTracker, log_classifier_metrics

    # Initialize tracker
    tracker = CometTracker(project_name="my-project", auto_login=True)

    # Start experiment
    tracker.start_experiment(
        experiment_name="robustness_test_20240101",
        tags=["pgd", "cifar10", "resnet"]
    )

    # Log parameters
    tracker.log_parameters({"lr": 0.001, "batch_size": 32})

    # Log metrics
    tracker.log_metrics({"accuracy": 95.3, "loss": 0.123})

    # Log files
    tracker.log_asset("results/output.csv")

    # End experiment
    tracker.end_experiment()
    ```
"""

import logging
from pathlib import Path
from typing import Any

import comet_ml


class CometTracker:
    """
    A wrapper class for Comet ML experiment tracking.

    Handles experiment initialization, logging, and cleanup with proper error handling.
    """

    def __init__(self, project_name: str = "rs-rd", auto_login: bool = True):
        """
        Initialize the CometTracker.

        Args:
            project_name: Name of the Comet ML project
            auto_login: Whether to automatically attempt Comet ML login
        """
        self.project_name = project_name
        self.experiment = None
        self._is_active = False

        if auto_login:
            self.login()

    def login(self) -> bool:
        """
        Attempt to log in to Comet ML.

        Returns:
            bool: True if login successful, False otherwise
        """
        try:
            comet_ml.login()
            logging.info("Comet ML login successful")
            return True
        except Exception as e:
            logging.warning(f"Comet ML login failed: {e}")
            return False

    def start_experiment(self, experiment_name: str, tags: list[str] | None = None) -> bool:
        """
        Start a new Comet ML experiment.

        Args:
            experiment_name: Name for the experiment
            tags: Optional list of tags for the experiment

        Returns:
            bool: True if experiment started successfully, False otherwise
        """
        if tags is None:
            tags = []

        experiment_config = comet_ml.ExperimentConfig(
            name=experiment_name,
            tags=tags,
        )

        try:
            self.experiment = comet_ml.start(
                project_name=self.project_name,
                experiment_config=experiment_config,
            )
            self._is_active = True
            logging.info(f"Comet ML experiment created: {self.experiment.url}")
            return True
        except Exception as e:
            logging.warning(f"Failed to create Comet ML experiment: {e}")
            logging.info("Continuing without Comet ML tracking...")
            self.experiment = None
            self._is_active = False
            return False

    def log_parameters(self, params: dict[str, Any]) -> bool:
        """
        Log experiment parameters to Comet ML.

        Args:
            params: Dictionary of parameters to log

        Returns:
            bool: True if logging successful, False otherwise
        """
        if not self._is_active or self.experiment is None:
            return False

        try:
            self.experiment.log_parameters(params)
            logging.info("Experiment parameters logged to Comet ML")
            return True
        except Exception as e:
            logging.warning(f"Failed to log parameters to Comet ML: {e}")
            return False

    def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> bool:
        """
        Log metrics to Comet ML.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time-series metrics

        Returns:
            bool: True if logging successful, False otherwise
        """
        if not self._is_active or self.experiment is None:
            return False

        try:
            self.experiment.log_metrics(metrics, step=step)
            return True
        except Exception as e:
            logging.warning(f"Failed to log metrics to Comet ML: {e}")
            return False

    def log_asset(self, file_path: str | Path, overwrite: bool = False) -> bool:
        """
        Log a file as an asset to Comet ML.

        Args:
            file_path: Path to the file to log
            overwrite: Whether to overwrite if asset already exists

        Returns:
            bool: True if logging successful, False otherwise
        """
        if not self._is_active or self.experiment is None:
            return False

        try:
            self.experiment.log_asset(str(file_path), overwrite=overwrite)
            logging.info(f"Asset logged to Comet ML: {file_path}")
            return True
        except Exception as e:
            logging.warning(f"Failed to log asset to Comet ML: {e}")
            return False

    def log_other(self, key: str, value: Any) -> bool:
        """
        Log other metadata to Comet ML.

        Args:
            key: Key for the metadata
            value: Value to log

        Returns:
            bool: True if logging successful, False otherwise
        """
        if not self._is_active or self.experiment is None:
            return False

        try:
            self.experiment.log_other(key, value)
            return True
        except Exception as e:
            logging.warning(f"Failed to log '{key}' to Comet ML: {e}")
            return False

    def end_experiment(self) -> bool:
        """
        End the Comet ML experiment.

        Returns:
            bool: True if experiment ended successfully, False otherwise
        """
        if not self._is_active or self.experiment is None:
            return False

        try:
            url = self.experiment.url
            self.experiment.end()
            self._is_active = False
            logging.info(f"Experiment completed. View results at: {url}")
            return True
        except Exception as e:
            logging.warning(f"Failed to end Comet ML experiment: {e}")
            return False

    @property
    def is_active(self) -> bool:
        """
        Check if experiment tracking is active.

        Returns:
            bool: True if experiment is active, False otherwise
        """
        return self._is_active

    @property
    def url(self) -> str | None:
        """
        Get the URL of the current experiment.

        Returns:
            str: URL of the experiment, or None if not active
        """
        if self.experiment:
            return self.experiment.url
        return None


def log_classifier_metrics(
    tracker: CometTracker,
    network_name: str,
    metrics: dict[str, Any],
) -> bool:
    """
    Log classifier performance metrics for a specific network.

    Args:
        tracker: CometTracker instance
        network_name: Name of the network
        metrics: Dictionary containing 'accuracy', 'correct', and 'total' keys

    Returns:
        bool: True if logging successful, False otherwise
    """
    if not tracker.is_active:
        return False

    metric_dict = {
        f"{network_name}_accuracy": metrics["accuracy"],
        f"{network_name}_correct_predictions": metrics["correct"],
        f"{network_name}_total_samples": metrics["total"],
    }

    success = tracker.log_metrics(metric_dict)
    if success:
        logging.info(f"Logged metrics for {network_name} to Comet ML")
    return success


def log_verona_experiment_summary(
    tracker: CometTracker,
    experiment_repository_path: Path,
    experiment_path: Path,
    results_path: Path,
    dataset_info: dict[str, Any],
    attack_info: dict[str, Any],
    epsilon_info: dict[str, Any],
) -> bool:
    """
    Log comprehensive VERONA experiment summary to Comet ML.

    Args:
        tracker: CometTracker instance
        experiment_repository_path: Path to experiment repository
        experiment_path: Path to experiment directory
        results_path: Path to results directory
        dataset_info: Dictionary with dataset information (name, split, size, etc.)
        attack_info: Dictionary with attack information (type, iterations, etc.)
        epsilon_info: Dictionary with epsilon configuration (start, stop, step, list)

    Returns:
        bool: True if logging successful, False otherwise
    """
    if not tracker.is_active:
        return False

    summary = {
        "dataset_name": dataset_info.get("name"),
        "split": dataset_info.get("split"),
        "sample_size": dataset_info.get("sample_size"),
        "sample_correct_predictions": dataset_info.get("sample_correct_predictions"),
        "experiment_repository_path": str(experiment_repository_path),
        "experiment_path": str(experiment_path),
        "results_path": str(results_path),
        "attack_type": attack_info.get("type"),
        "attack_iterations": attack_info.get("iterations"),
        "epsilon_range": f"{epsilon_info.get('start')} to {epsilon_info.get('stop')} (step {epsilon_info.get('step')})",
        "total_epsilon_values": len(epsilon_info.get("list", [])),
    }

    success = tracker.log_other("experiment_summary", summary)
    if success:
        logging.info("Experiment summary logged to Comet ML")
    return success


def log_verona_results(
    tracker: CometTracker,
    results_path: Path,
) -> bool:
    """
    Log VERONA result files to Comet ML.

    Logs the main result_df.csv, per_epsilon_results.csv, and configuration.json if they exist.

    Args:
        tracker: CometTracker instance
        results_path: Path to the results directory

    Returns:
        bool: True if at least one file was logged successfully, False otherwise
    """
    if not tracker.is_active:
        return False

    success_count = 0

    # Log result dataframe
    result_csv = results_path / "result_df.csv"
    if result_csv.exists():
        if tracker.log_asset(result_csv):
            logging.info(f"Result CSV logged to Comet ML: {result_csv}")
            success_count += 1

    # Log per-epsilon results
    per_epsilon_csv = results_path / "per_epsilon_results.csv"
    if per_epsilon_csv.exists():
        if tracker.log_asset(per_epsilon_csv):
            logging.info(f"Per-epsilon CSV logged to Comet ML: {per_epsilon_csv}")
            success_count += 1

    # Log configuration
    experiment_path = results_path.parent
    config_file = experiment_path / "configuration.json"
    if config_file.exists():
        if tracker.log_asset(config_file):
            logging.info(f"Configuration file logged to Comet ML: {config_file}")
            success_count += 1

    return success_count > 0

