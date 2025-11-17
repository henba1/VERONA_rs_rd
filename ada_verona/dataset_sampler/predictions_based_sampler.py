import logging
from typing import Any

import torch

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.machine_learning_model.network import Network
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler


class PredictionsBasedSampler(DatasetSampler):
    """
    A sampler class that selects data points based on the predictions of a network.
    """

    def __init__(self, sample_correct_predictions: bool = True, top_k: int = 1) -> None:
        """
        Initialize the PredictionsBasedSampler with the given parameter.

        Args:
            sample_correct_predictions (bool, optional): Whether to sample data points with correct predictions.
            Defaults to True as in the JAIR paper.
            top_k: Number of top scores to take into account for checking the correct prediction.
        """
        self.sample_correct_predictions = sample_correct_predictions
        self.top_k = top_k

    def sample(self, network: Network, dataset: ExperimentDataset) -> ExperimentDataset:
        """
        Sample data points from the dataset based on the predictions of the network.

        Args:
            network (Network): The network to use for predictions.
            dataset (ExperimentDataset): The dataset to sample from.

        Returns:
            ExperimentDataset: The sampled dataset.
        """

        selected_indices = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = network.load_pytorch_model().to(device)
        model.eval()

        for data_point in dataset:
            data = data_point.data.reshape(network.get_input_shape())
            data = data.to(device)

            with torch.no_grad():
                output = model(data)

            _, predicted_labels = torch.topk(output, self.top_k)
            predicted_labels = predicted_labels.cpu()

            if self.sample_correct_predictions:
                if int(data_point.label) in predicted_labels:
                    selected_indices.append(data_point.id)
            else:
                if int(data_point.label) not in predicted_labels:
                    selected_indices.append(data_point.id)

        return dataset.get_subset(selected_indices)

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
            data = data_point.data.reshape(network.get_input_shape())  # Reshape data to network input shape
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
