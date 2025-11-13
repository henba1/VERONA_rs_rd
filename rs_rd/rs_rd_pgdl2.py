import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import comet_ml
import numpy as np

# experiment utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiment_utils import create_distribution, get_balanced_sample
from rs_rd_research.paths import get_dataset_dir, get_models_dir, get_results_dir

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

# Set up logging using the custom logger
logger.setup_logging(level=logging.INFO)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Log experiment start time
    start_time = time.time()

    # ---------------------------------------Basic Experiment Settings -----------------------------------------
    dataset_name = "CIFAR-10"
    split = "test"
    sample_size = 500
    random_seed = 42
    sample_correct_predictions = True

    experiment_type = "verona_upper_bounding"
    experiment_name = "pgd_l2"

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 0.4
    epsilon_step = 0.0039
    # ----------------------------------------DATASET AND MODELS DIRECTORY CONFIGURATION---------------------------
    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name) / experiment_type
    RESULTS_DIR = get_results_dir()

    # --------- -----------------------------COMET initialization ------------------------------------------------
    try:
        comet_ml.login()
        logging.info("Comet ML login successful")
    except Exception as e:
        logging.warning(f"Comet ML login failed: {e}")

    # Create experiment config with descriptive name
    experiment_config = comet_ml.ExperimentConfig(
        name=f"rs_rd_pgd_l2_CIFAR-10_{timestamp}",
        tags=["rs-rd", f"{experiment_type}", f"{dataset_name}", f"{experiment_name}"],
    )

    try:
        experiment = comet_ml.start(
            project_name="rs-rd",
            experiment_config=experiment_config,
        )
        logging.info(f"Comet ML experiment created: {experiment.url}")
    except Exception as e:
        logging.warning(f"Failed to create Comet ML experiment: {e}")
        logging.info("Continuing without Comet ML tracking...")
        experiment = None

    # ----------------------------------------EXPERIMENT REPOSITORY CONFIGURATION----------------------------------
    experiment_repository_path = (
        Path(RESULTS_DIR)
        / f"verona_rs_rd_{experiment_name}_{dataset_name}_{sample_size}_sample_correct_{sample_correct_predictions}"
    )
    os.makedirs(experiment_repository_path, exist_ok=True)
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=MODELS_DIR)
    experiment_repository.initialize_new_experiment(experiment_name)

    # Log experiment parameters to Comet ML
    if experiment:
        try:
            experiment.log_parameters(
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "sample_size": sample_size,
                    "sample_correct_predictions": sample_correct_predictions,
                    "experiment_type": experiment_type,
                    "experiment_name": experiment_name,
                    "attack_type": "PGD",
                    "attack_norm": "l2",
                    "attack_iterations": 40,
                    "epsilon_start": epsilon_start,
                    "epsilon_stop": epsilon_stop,
                    "epsilon_step": epsilon_step,
                    "sampling_method": "StratifiedShuffleSplit",
                    "random_seed_dataset_sampling": random_seed,
                }
            )
            logging.info("Experiment parameters logged to Comet ML")
        except Exception as e:
            logging.warning(f"Failed to log parameters to Comet ML: {e}")

    # ----------------------------------------DATASET CONFIGURATION-----------------------------------------------
    # load dataset and get original CIFAR-10 indices
    cifar10_torch_dataset, original_indices = get_balanced_sample(
        dataset_name=dataset_name,
        train_bool=(split == "train"),
        dataset_size=sample_size,
        dataset_dir=DATASET_DIR,
        seed=random_seed,
    )
    dataset = PytorchExperimentDataset(dataset=cifar10_torch_dataset)

    # ----------------------------------------SAVE ORIGINAL DATASET INDICES----------------------------------------
    indices_file = (
        experiment_repository.get_act_experiment_path()
        / f"original_{dataset_name}_indices_{split}_nsample_{sample_size}_{timestamp}.txt"
    )

    np.savetxt(
        indices_file, original_indices, fmt="%d", header=f"Original CIFAR-10 {split} indices for balanced sample"
    )
    logging.info(f"Saved original {dataset_name} indices to {indices_file}")

    # Log indices file to Comet ML
    if experiment:
        try:
            experiment.log_asset(str(indices_file))
            logging.info(f"Indices file logged to Comet ML: {indices_file}")
        except Exception as e:
            logging.warning(f"Failed to log indices file to Comet ML: {e}")

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 0.4
    epsilon_step = 0.0039
    epsilon_list = np.arange(epsilon_start, epsilon_stop, epsilon_step)

    # ----------------------------------------DATASET SAMPLER CONFIGURATION------------------------------------------
    # only sample correct predictions
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=sample_correct_predictions)

    # ----------------------------------------VERIFICATION CONFIGURATION---------------------------------------------
    # 10, 0, 1 (default) for CIFAR-10, same for MNIST #TODO adjust for ImageNet
    property_generator = One2AnyPropertyGenerator()
    robustness_attack_estimator = AttackEstimationModule(attack=PGDAttack(number_iterations=40, norm="l2"))

    # ----------------------------------------EPSILON VALUE SEARCH CONFIGURATION-------------------------------------
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=robustness_attack_estimator
    )

    # ----------------------------------------CREATE ROBUSTNESS DISTRIBUTION------------------------------------------
    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator, property_generator)

    # ----------------------------------------LOG RESULTS TO COMET ML---------------------------------------------
    experiment_path = experiment_repository.get_act_experiment_path()
    results_path = experiment_repository.get_results_path()

    if experiment:
        try:
            # Log experiment metadata
            experiment.log_other(
                "experiment_summary",
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "sample_size": sample_size,
                    "sample_correct_predictions": sample_correct_predictions,
                    "experiment_repository_path": str(experiment_repository_path),
                    "experiment_path": str(experiment_path),
                    "results_path": str(results_path),
                    "attack_type": "PGD (L2)",
                    "attack_iterations": 40,
                    "epsilon_range": f"{epsilon_start} to {epsilon_stop} (step {epsilon_step})",
                    "total_epsilon_values": len(epsilon_list),
                },
            )
            logging.info("Experiment summary logged to Comet ML")
        except Exception as e:
            logging.warning(f"Failed to log experiment summary to Comet ML: {e}")

        try:
            # Log result dataframe if it exists
            result_csv = results_path / "result_df.csv"
            if result_csv.exists():
                experiment.log_asset(str(result_csv))
                logging.info(f"Result CSV logged to Comet ML: {result_csv}")
        except Exception as e:
            logging.warning(f"Failed to log result CSV to Comet ML: {e}")

        try:
            # Log per-epsilon results if they exist
            per_epsilon_csv = results_path / "per_epsilon_results.csv"
            if per_epsilon_csv.exists():
                experiment.log_asset(str(per_epsilon_csv))
                logging.info(f"Per-epsilon CSV logged to Comet ML: {per_epsilon_csv}")
        except Exception as e:
            logging.warning(f"Failed to log per-epsilon CSV to Comet ML: {e}")

        try:
            # Log configuration if it exists
            config_file = experiment_path / "configuration.json"
            if config_file.exists():
                experiment.log_asset(str(config_file))
                logging.info(f"Configuration file logged to Comet ML: {config_file}")
        except Exception as e:
            logging.warning(f"Failed to log configuration to Comet ML: {e}")

        # Log final metrics
        try:
            experiment.log_metrics(
                {
                    "total_duration_seconds": time.time() - start_time,
                }
            )
            logging.info("Final metrics logged to Comet ML")
        except Exception as e:
            logging.warning(f"Failed to log final metrics to Comet ML: {e}")

        # End the experiment
        try:
            experiment.end()
            logging.info(f"Experiment completed. View results at: {experiment.url}")
        except Exception as e:
            logging.warning(f"Failed to end Comet ML experiment: {e}")
    else:
        logging.info("Experiment completed (Comet ML tracking was not available)")


if __name__ == "__main__":
    main()
