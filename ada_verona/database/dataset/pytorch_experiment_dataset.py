# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from torch.utils.data import Dataset
from typing_extensions import Self

from ada_verona.database.dataset.data_point import DataPoint


class PytorchExperimentDataset:
    """
    A dataset class for wrapping a PyTorch dataset for experiments.
    """

    def __init__(self, dataset: Dataset, original_indices: list[int] | None = None) -> None:
        """
        Initialize the PytorchExperimentDataset with a PyTorch dataset.

        Args:
            dataset (Dataset): The PyTorch dataset to wrap.
            original_indices (list[int] | None): Optional list mapping local indices to original dataset indices.
                If None, uses sequential indices [0, 1, 2, ...].
        """
        self.dataset = dataset
        self._indices = [x for x in range(0, len(dataset))]
        # Store mapping from local index to original dataset index
        self._original_indices = original_indices if original_indices is not None else list(self._indices)

    def __len__(self) -> int:
        """
        Get the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self._indices)

    def __getitem__(self, idx) -> DataPoint:
        """
        Get the data point at the specified index.

        Args:
            idx (int): The index of the data point.

        Returns:
            DataPoint: The data point at the specified index.
        """
        index = self._indices[idx]
        # Use original index as the DataPoint ID
        original_index = self._original_indices[index]

        data, label = self.dataset[index]

        return DataPoint(original_index, label, data)

    def get_subset(self, indices: list[int]) -> Self:
        """
        Get a subset of the underlying pytorch dataset for
        the specified indices.

        Args:
            indices (list[int]): The list of indices to get the subset for.
                These should be original dataset indices.

        Returns:
            Self: The subset of the dataset.
        """
        new_instance = PytorchExperimentDataset(self.dataset, original_indices=self._original_indices)

        # Map original indices back to local indices
        # Create a reverse mapping: original_index -> local_index
        original_to_local = {orig: local for local, orig in enumerate(self._original_indices)}

        # Convert provided original indices to local indices
        new_instance._indices = [original_to_local[orig_idx] for orig_idx in indices if orig_idx in original_to_local]

        return new_instance

    def __str__(self) -> str:
        """
        Get the string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return str(self._indices)
