"""
PGD L2 robustness experiment for CIFAR-10 with ONNX models.

Loads ONNX models from models/CIFAR-10/RS_Baseline/. Uses same attack settings:
40 iterations, L2 norm for [0,1] pixel space.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import ada_verona.util.logger as logger
from ada_verona import (
    AttackEstimationModule,
    BinarySearchEpsilonValueEstimator,
    CometTracker,
    ExperimentRepository,
    IdentitySampler,
    One2AnyPropertyGenerator,
    PGDAttack,
    PredictionsBasedSampler,
    PytorchExperimentDataset,
    create_distribution,
    get_balanced_sample,
    get_dataset_dir,
    get_first_n,
    get_models_dir,
    get_results_dir,
    get_sample,
    log_classifier_metrics,
    log_verona_experiment_summary,
    log_verona_results,
)

logger.setup_logging(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib").setLevel(logging.WARNING)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    # ---------------------------------------Basic Experiment Settings -----------------------------------------
    dataset_name = "CIFAR-10"
    split = "test"
    sample_size = 300
    random_seed = 42
    use_first_n = True
    experiment_tag = None
    use_identity_sampler = False
    sample_correct_predictions = True
    sample_stratified = False

    experiment_type = "RS_Baseline"
    experiment_name = "pgd_l2"

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 2
    epsilon_step = 1 / 255
    epsilon_list = np.arange(epsilon_start, epsilon_stop, epsilon_step)

    # ----------------------------------------DATASET AND MODELS DIRECTORY CONFIGURATION---------------------------
    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name) / experiment_type
    RESULTS_DIR = get_results_dir()

    comet_tracker = CometTracker(project_name="rs-rd", auto_login=True)

    # ----------------------------------------EXPERIMENT REPOSITORY CONFIGURATION----------------------------------
    experiment_dir_name = (
        f"verona_rs_rd_{experiment_name}_{dataset_name}_{sample_size}_"
        f"sample_correct_{sample_correct_predictions}_"
        f"sample_stratified_{sample_stratified}"
    )
    experiment_repository_path = Path(RESULTS_DIR) / experiment_dir_name
    os.makedirs(experiment_repository_path, exist_ok=True)
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=MODELS_DIR)

    network_list = experiment_repository.get_network_list()

    if len(network_list) == 0:
        logging.error(f"No ONNX models found in {MODELS_DIR}. Ensure .onnx files exist.")
        return

    model_names = [network.name for network in network_list]

    epsilon_tag = f"eps_{epsilon_start}_{epsilon_stop}_{epsilon_step}"
    comet_tracker.start_experiment(
        experiment_name=f"rs_rd_pgd_l2_CIFAR-10_onnx_{timestamp}",
        tags=[experiment_type, dataset_name, experiment_name, epsilon_tag, *model_names],
        experiment_tag=experiment_tag,
    )
    experiment_repository.initialize_new_experiment(experiment_name)

    comet_tracker.log_parameters(
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
            "random_seed_dataset_sampling": random_seed,
        }
    )

    # ----------------------------------------DATASET CONFIGURATION-----------------------------------------------
    # Standard CIFAR-10 for ONNX: 32x32, flatten=True, [0,1] range
    image_size = None
    flatten = True

    if use_first_n:
        torch_dataset, original_indices = get_first_n(
            dataset_name=dataset_name,
            train_bool=(split == "train"),
            n=sample_size,
            dataset_dir=DATASET_DIR,
            image_size=image_size,
            flatten=flatten,
        )
    elif sample_stratified:
        torch_dataset, original_indices = get_balanced_sample(
            dataset_name=dataset_name,
            train_bool=(split == "train"),
            dataset_size=sample_size,
            dataset_dir=DATASET_DIR,
            seed=random_seed,
            image_size=image_size,
            flatten=flatten,
        )
    else:
        torch_dataset, original_indices = get_sample(
            dataset_name=dataset_name,
            train_bool=(split == "train"),
            dataset_size=sample_size,
            dataset_dir=DATASET_DIR,
            seed=random_seed,
            image_size=image_size,
            flatten=flatten,
        )

    dataset = PytorchExperimentDataset(dataset=torch_dataset, original_indices=original_indices.tolist())

    # ----------------------------------------SAVE ORIGINAL DATASET INDICES----------------------------------------
    indices_file = (
        experiment_repository.get_act_experiment_path()
        / f"original_{dataset_name}_indices_{split}_nsample_{sample_size}_{timestamp}.txt"
    )
    np.savetxt(
        indices_file,
        original_indices,
        fmt="%d",
        header=f"Original CIFAR-10 {split} indices (n_sample={sample_size})",
    )
    logging.info(f"Saved original {dataset_name} indices to {indices_file}")
    comet_tracker.log_asset(indices_file)

    # ----------------------------------------DATASET SAMPLER CONFIGURATION------------------------------------------
    if use_identity_sampler:
        dataset_sampler = IdentitySampler()
    else:
        dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=sample_correct_predictions)

    # ----------------------------------------CLASSIFIER PERFORMANCE METRICS-----------------------------------------
    logging.info(f"Computing classifier metrics for {len(network_list)} network(s)")

    for network in network_list:
        try:
            metrics = dataset_sampler.compute_metrics(network, dataset)
            log_classifier_metrics(comet_tracker, network.name, metrics)

            predictions_file = (
                experiment_repository.get_act_experiment_path() / f"{network.name}_predictions_{timestamp}.txt"
            )
            with open(predictions_file, "w") as f:
                f.write(f"Network: {network.name}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Correct: {metrics['correct']}/{metrics['total']}\n\n")
                f.write("Sample_ID,True_Label,Predicted_Label,Correct\n")
                for i, (true_label, pred_label) in enumerate(
                    zip(metrics["labels"], metrics["predictions"], strict=True)
                ):
                    is_correct = "Yes" if true_label == pred_label else "No"
                    f.write(f"{i},{true_label},{pred_label},{is_correct}\n")

            logging.info(f"Saved metrics to {predictions_file}")
            comet_tracker.log_asset(predictions_file)

        except Exception as e:
            logging.error(f"Failed to compute metrics for network {network.name}: {e}")

    # ----------------------------------------VERIFICATION CONFIGURATION---------------------------------------------
    property_generator = One2AnyPropertyGenerator()
    robustness_attack_estimator = AttackEstimationModule(attack=PGDAttack(number_iterations=40, norm="l2"))

    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=robustness_attack_estimator
    )

    # ----------------------------------------CREATE ROBUSTNESS DISTRIBUTION------------------------------------------
    create_distribution(
        experiment_repository,
        dataset,
        dataset_sampler,
        epsilon_value_estimator,
        property_generator,
        network_list=network_list,
    )
    results_path = experiment_repository.get_results_path()

    log_verona_results(comet_tracker, results_path)
    logging.info("Result files contain original dataset indices in the image_id column")

    # ----------------------------------------LOG RESULTS TO COMET ML---------------------------------------------
    experiment_path = experiment_repository.get_act_experiment_path()
    log_verona_experiment_summary(
        comet_tracker,
        experiment_repository_path=experiment_repository_path,
        experiment_path=experiment_path,
        results_path=results_path,
        dataset_info={
            "name": dataset_name,
            "split": split,
            "sample_size": sample_size,
            "sample_correct_predictions": sample_correct_predictions,
        },
        attack_info={
            "type": experiment_name,
            "iterations": 40,
        },
        epsilon_info={
            "start": epsilon_start,
            "stop": epsilon_stop,
            "step": epsilon_step,
            "list": epsilon_list,
        },
    )

    comet_tracker.log_metrics({"total_duration_seconds": time.time() - start_time})
    comet_tracker.end_experiment()


if __name__ == "__main__":
    main()
