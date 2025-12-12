import contextlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from foolbox.attacks import L2ProjectedGradientDescentAttack

import ada_verona.util.logger as logger
from ada_verona import (
    AttackEstimationModule,
    BinarySearchEpsilonValueEstimator,
    CometTracker,
    ExperimentRepository,
    FoolboxAttack,
    IdentitySampler,
    One2AnyPropertyGenerator,
    PGDAttack,
    PredictionsBasedSampler,
    PytorchExperimentDataset,
    create_distribution,
    get_balanced_sample,
    get_dataset_dir,
    get_models_dir,
    get_results_dir,
    get_sample,
    load_networks_from_directory,
    log_classifier_metrics,
    log_verona_experiment_summary,
    log_verona_results,
)

logger.setup_logging(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("dulwich").setLevel(logging.WARNING)
logging.getLogger("comet_ml").setLevel(logging.INFO)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Log experiment start time
    start_time = time.time()

    # ---------------------------------------Basic Experiment Settings -----------------------------------------
    dataset_name = "CIFAR-10"
    input_shape = (1, 3, 32, 32)
    split = "test"
    sample_size = 5
    random_seed = 42
    # If want full dataset, use IdentitySampler, otherwise use PredictionsBasedSampler
    use_identity_sampler = False
    sample_correct_predictions = True
    sample_stratified = False

    experiment_type = "experiment_refactor_test"  # Experiment type for tracking
    models_experiment_type = "experiment_refactor_test"  # Directory to load models from

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 0.5
    epsilon_step = 8 / 255
    # ----------------------------------------DATASET AND MODELS DIRECTORY CONFIGURATION---------------------------
    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name) / models_experiment_type  # Load models from sdp_verification directory
    RESULTS_DIR = get_results_dir()

    # --------- -----------------------------COMET ML TRACKING INITIALIZATION --------------------------------
    comet_tracker = CometTracker(project_name="rs-rd", auto_login=True)

    # ----------------------------------------EXPERIMENT REPOSITORY CONFIGURATION----------------------------------
    experiment_dir_name = (
        f"verona_rs_rd_attacks_{dataset_name}_{sample_size}_"
        f"sample_correct_{sample_correct_predictions}_"
        f"sample_stratified_{sample_stratified}"
    )
    experiment_repository_path = Path(RESULTS_DIR) / experiment_dir_name
    os.makedirs(experiment_repository_path, exist_ok=True)
    # networks are loaded from this object
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=MODELS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_list = load_networks_from_directory(MODELS_DIR, input_shape, device)
    model_names = [network.name for network in network_list]

    if len(network_list) == 0:
        logging.error(f"No models found in {MODELS_DIR}")
        logging.error("Ensure models exist in models directory (formats: .onnx, .pth)")
        return

    epsilon_tag = f"eps_{epsilon_start}_{epsilon_stop}_{epsilon_step}"
    epsilon_list = np.arange(epsilon_start, epsilon_stop, epsilon_step)

    # Log experiment parameters to Comet ML (will be logged per attack type)
    base_params = {
        "dataset_name": dataset_name,
        "split": split,
        "sample_size": sample_size,
        "use_identity_sampler": use_identity_sampler,
        "sample_correct_predictions": sample_correct_predictions if not use_identity_sampler else None,
        "experiment_type": experiment_type,
        "epsilon_start": epsilon_start,
        "epsilon_stop": epsilon_stop,
        "epsilon_step": epsilon_step,
        "get_sample_method": "Stratified" if sample_stratified else "Non-stratified",
        "random_seed_dataset_sampling": random_seed,
    }

    # ----------------------------------------DATASET CONFIGURATION-----------------------------------------------
    # load dataset and get original CIFAR-10 indices
    if sample_stratified:
        cifar10_torch_dataset, original_indices = get_balanced_sample(
            dataset_name=dataset_name,
            train_bool=(split == "train"),
            dataset_size=sample_size,
            dataset_dir=DATASET_DIR,
            seed=random_seed,
        )
    else:
        cifar10_torch_dataset, original_indices = get_sample(
            dataset_name=dataset_name,
            train_bool=(split == "train"),
            dataset_size=sample_size,
            dataset_dir=DATASET_DIR,
            seed=random_seed,
        )
    # Pass original_indices to maintain mapping to original dataset
    dataset = PytorchExperimentDataset(dataset=cifar10_torch_dataset, original_indices=original_indices.tolist())

    # ----------------------------------------VERIFICATION CONFIGURATION---------------------------------------------
    # 10, 0, 1 (default) for CIFAR-10, same for MNIST #TODO adjust for ImageNet
    property_generator = One2AnyPropertyGenerator()

    # Define multiple attack configurations
    attack_configs = [
        {
            "name": "pgd_l2",
            "attack": PGDAttack(number_iterations=40, norm="l2"),
            "attack_type": "PGD",
            "attack_iterations": 40,
        },
        # {
        #     "name": "foolbox_pgd_l2",
        #     "attack": FoolboxAttack(L2ProjectedGradientDescentAttack, bounds=(0, 1), steps=40),
        #     "attack_type": "Foolbox PGD L2",
        #     "attack_iterations": 40,   #NOTE: Foolbox attacks are not supported for onnx models
        # },
        # {
        #     "name": "foolbox_cw_l2",
        #     "attack": FoolboxAttack(L2CarliniWagnerAttack, bounds=(0, 1), steps=100),
        #     "attack_type": "Foolbox CW L2",
        #     "attack_iterations": 100,
        # },
    ]

    # Initialize first experiment for classifier metrics computation and indices file saving
    # (metrics are computed once and shared across all attacks)
    first_attack_name = attack_configs[0]["name"]
    experiment_repository.initialize_new_experiment(first_attack_name)

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
    comet_tracker.log_asset(indices_file)

    # ----------------------------------------DATASET SAMPLER CONFIGURATION------------------------------------------
    if use_identity_sampler:
        dataset_sampler = IdentitySampler()
    else:
        dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=sample_correct_predictions)

    # ----------------------------------------CLASSIFIER PERFORMANCE METRICS-----------------------------------------
    # Compute classifier metrics for each network before running robustness verification
    # Metrics are computed once and logged for each attack experiment
    logging.info(f"Computing classifier metrics for {len(network_list)} network(s)")
    network_metrics = {}

    for network in network_list:
        try:
            # Compute metrics on full dataset (before sampling) #TODO: kinda pointless for PredictionsBasedSampler
            metrics = dataset_sampler.compute_metrics(network, dataset)
            network_metrics[network.name] = metrics

            # Log detailed predictions to file
            predictions_file = (
                experiment_repository.get_act_experiment_path() / f"{network.name}_predictions_{timestamp}.txt"
            )
            with open(predictions_file, "w") as f:
                f.write(f"Network: {network.name}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Correct: {metrics['correct']}/{metrics['total']}\n\n")
                f.write("Sample_ID,True_Label,Predicted_Label,Correct\n")
                zipped_preds = zip(metrics["labels"], metrics["predictions"], strict=True)
                for i, (true_label, pred_label) in enumerate(zipped_preds):
                    is_correct = "Yes" if true_label == pred_label else "No"
                    f.write(f"{i},{true_label},{pred_label},{is_correct}\n")

            logging.info(f"Saved metrics to {predictions_file}")

        except Exception as e:
            logging.error(f"Failed to compute metrics for network {network.name}: {e}")

    # Run each attack configuration
    logging.info(f"Running {len(attack_configs)} attack configuration(s): {[cfg['name'] for cfg in attack_configs]}")
    for idx, attack_config in enumerate(attack_configs):
        attack_name = attack_config["name"]
        logging.info(f"Starting experiment for attack: {attack_name} (attack {idx + 1}/{len(attack_configs)})")

        try:
            if idx > 0 or attack_name != first_attack_name:
                experiment_repository.initialize_new_experiment(attack_name)

            # Start Comet experiment for this attack
            comet_tracker.start_experiment(
                experiment_name=f"rs_rd_{attack_name}_CIFAR-10_{timestamp}",
                tags=[experiment_type, dataset_name, attack_name, epsilon_tag, *model_names],
            )

            # Log attack-specific parameters
            attack_params = {
                **base_params,
                "experiment_name": attack_name,
                "attack_type": attack_config["attack_type"],
                "attack_iterations": attack_config["attack_iterations"],
            }
            comet_tracker.log_parameters(attack_params)

            # Log classifier metrics for this attack experiment (using pre-computed metrics)
            for network in network_list:
                if network.name in network_metrics:
                    log_classifier_metrics(comet_tracker, network.name, network_metrics[network.name])
                    # Log predictions file if this is the first attack
                    if idx == 0:
                        predictions_file = (
                            experiment_repository.get_act_experiment_path()
                            / f"{network.name}_predictions_{timestamp}.txt"
                        )
                        if predictions_file.exists():
                            comet_tracker.log_asset(predictions_file)

            # Create robustness estimator
            robustness_attack_estimator = AttackEstimationModule(attack=attack_config["attack"])

            # Create epsilon value estimator with this attack estimator
            epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
                epsilon_value_list=epsilon_list.copy(), verifier=robustness_attack_estimator
            )

            # Create robustness distribution
            create_distribution(
                experiment_repository,
                dataset,
                dataset_sampler,
                epsilon_value_estimator,
                property_generator,
                network_list=network_list,
            )
            results_path = experiment_repository.get_results_path()

            # Log result files (image_id column now contains original CIFAR-10 indices)
            log_verona_results(comet_tracker, results_path)
            logging.info("Result files contain original dataset indices in the image_id column")

            # Log experiment summary
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
                    "type": attack_name,
                },
                epsilon_info={
                    "start": epsilon_start,
                    "stop": epsilon_stop,
                    "step": epsilon_step,
                    "list": epsilon_list,
                },
            )

            comet_tracker.end_experiment()
            logging.info(f"Completed experiment for attack: {attack_name}")

        except Exception as e:
            logging.error(f"Failed to complete experiment for attack {attack_name}: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                comet_tracker.end_experiment()
            logging.warning(f"Skipping to next attack after failure in {attack_name}")
            continue

    # Log final metrics
    total_duration = time.time() - start_time
    logging.info(f"Total duration for all attacks: {total_duration:.2f} seconds")


if __name__ == "__main__":
    main()
