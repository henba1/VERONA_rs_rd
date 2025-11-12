import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import ada_verona.util.logger as logger
from ada_verona import (
    AttackEstimationModule,
    BinarySearchEpsilonValueEstimator,
    ExperimentRepository,
    One2AnyPropertyGenerator,
    PGDAttack,
    PredictionsBasedSampler,
    PytorchExperimentDataset,
)
from experiment_utils import create_distribution, get_balanced_sample
from rs_rd_research_code.paths import get_dataset_dir, get_models_dir, get_results_dir

# Set up logging using the custom logger
logger.setup_logging(level=logging.INFO)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ---------------------------------------Basic Experiment Settings -----------------------------------------
    dataset_name = "CIFAR-10"
    split = "test"
    sample_size = 500

    sample_correct_predictions = True

    experiment_name = f"pgd_l2_{dataset_name}_{split}_nsample_{sample_size}_{sample_correct_predictions}"

    DATASET_DIR = get_dataset_dir(dataset_name) 
    MODELS_DIR = get_models_dir(dataset_name) 
    RESULTS_DIR = get_results_dir()

    # ----------------------------------------EXPERIMENT REPOSITORY CONFIGURATION----------------------------------
    experiment_repository_path = (
        Path(RESULTS_DIR) / f"{dataset_name}" / f"verona_rs_rd_{experiment_name}_{dataset_name}_{sample_size}"
    )
    os.makedirs(experiment_repository_path, exist_ok=True)
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=MODELS_DIR)
    experiment_repository.initialize_new_experiment(experiment_name)

    # ----------------------------------------DATASET CONFIGURATION-----------------------------------------------
    # load dataset and get original CIFAR-10 indices
    cifar10_torch_dataset, original_indices = get_balanced_sample(
        dataset_name=dataset_name, train_bool=(split == "train"), dataset_size=sample_size, dataset_dir=DATASET_DIR
    )
    dataset = PytorchExperimentDataset(dataset=cifar10_torch_dataset)

    # ----------------------------------------SAVE ORIGINAL DATASET INDICES----------------------------------------
    indices_file = (
        experiment_repository_path / f"original_{dataset_name}_indices_{split}_nsample_{sample_size}_{timestamp}.txt"
    )

    np.savetxt(
        indices_file, original_indices, fmt="%d", header=f"Original CIFAR-10 {split} indices for balanced sample"
    )
    logging.info(f"Saved original {dataset_name} indices to {indices_file}")

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_list = np.arange(0.00, 0.4, 0.0039)

    # ----------------------------------------DATASET SAMPLER CONFIGURATION------------------------------------------
    # only sample correct predictions
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=sample_correct_predictions)

    # ----------------------------------------VERIFICATION CONFIGURATION---------------------------------------------
    property_generator = One2AnyPropertyGenerator()  # 10, 0, 1 (default) for CIFAR-10, same for MNIST #TODO  adjust for ImageNet
    robustness_attack_estimator = AttackEstimationModule(attack=PGDAttack(number_iterations=40, norm="l2"))

    # ----------------------------------------EPSILON VALUE SEARCH CONFIGURATION-------------------------------------
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=robustness_attack_estimator
    )

    # ----------------------------------------CREATE ROBUSTNESS DISTRIBUTION------------------------------------------
    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator, property_generator)


if __name__ == "__main__":
    main()