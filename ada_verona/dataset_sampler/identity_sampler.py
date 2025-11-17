import logging
from typing import Any

import torch

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.machine_learning_model.network import Network
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler


class IdentitySampler(DatasetSampler):
    """
    A sampler class that returns the entire dataset without any filtering.

    This is useful when you want to process all data points without any selection criteria.
    """

    def sample(self, network: Network, dataset: ExperimentDataset) -> ExperimentDataset:
        """
        Return the entire dataset without sampling.

        Args:
            network (Network): The network (not used, but required by interface).
            dataset (ExperimentDataset): The dataset to sample from.

        Returns:
            ExperimentDataset: The complete dataset.
        """
        return dataset

    def compute_metrics(self, network: Network, dataset: ExperimentDataset) -> dict[str, Any]:
        """Compute classifier performance metrics on the dataset.

        Args:
            network: VERONA Network object
            dataset: ExperimentDataset

        Returns:
            dict: Dictionary with metrics:
                - accuracy: Percentage of correct predictions
                - correct: Number of correctly classified samples
                - total: Total number of samples
                - predictions: List of predicted labels
                - labels: List of true labels
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = network.load_pytorch_model().to(device)
        model.eval()

        correct = 0
        total = 0
        predictions = []
        labels = []

        logging.info(f"Computing classifier metrics for network: {network.name}")

        for data_point in dataset:
            # Reshape data to network input shape
            data = data_point.data.reshape(network.get_input_shape())
            data = data.to(device)

            with torch.no_grad():
                output = model(data)

            _, predicted = torch.max(output, 1)
            pred_label = predicted.cpu().item()
            true_label = data_point.label

            predictions.append(pred_label)
            labels.append(true_label)

            if pred_label == true_label:
                correct += 1
            total += 1

        accuracy = 100 * correct / total if total > 0 else 0

        logging.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions,
            "labels": labels,
        }
