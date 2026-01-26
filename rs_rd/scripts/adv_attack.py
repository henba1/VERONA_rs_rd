import contextlib
import logging
import time
from datetime import datetime

import numpy as np
import torch

import ada_verona.util.logger as logger
from ada_verona import (
    AttackEstimationModule,
    BinarySearchEpsilonValueEstimator,
    CometTracker,
    ExperimentRepository,
    IdentitySampler,
    One2AnyPropertyGenerator,
    PredictionsBasedSampler,
    PytorchExperimentDataset,
    RestartingPGDAttack,
    create_distribution,
    create_experiment_directory,
    get_balanced_sample,
    get_dataset_dir,
    get_models_dir,
    get_results_dir,
    get_sample,
    log_classifier_metrics,
    log_verona_experiment_summary,
    log_verona_results,
    save_original_indices,
    sdp_crown_models_loading,
)

logger.setup_logging(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("dulwich").setLevel(logging.WARNING)
logging.getLogger("comet_ml").setLevel(logging.INFO)


# use sdp_crown specific preprocessing for CIFAR-10
def main():
    start_time = time.time()

    # ---------------------------------------BASIC EXPERIMENT CONFIGURATION -----------------------------------------
    experiment_type = "conv_large_vs_conv_large_adv"
    dataset_name = "CIFAR-10"
    input_shape = (1, 3, 32, 32)
    split = "test"
    sample_size = 1000
    random_seed = 5432
    # Optional prefix used for grouping multiple runs (directory name + Comet tags).
    # Keep this different from `experiment_type` to avoid duplicated prefixes.
    experiment_tag = None

    use_identity_sampler = False
    sample_correct_predictions = True
    sample_stratified = False

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 5
    epsilon_step = 0.00784314
    # ----------------------------------------DATASET AND MODELS DIRECTORY CONFIGURATION---------------------------
    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name) / "conv_large_sdp_compar"
    RESULTS_DIR = get_results_dir()

    # --------- -----------------------------COMET ML TRACKING INITIALIZATION --------------------------------
    comet_tracker = CometTracker(project_name="rs-rd", auto_login=True)

    # ----------------------------------------EXPERIMENT REPOSITORY CONFIGURATION----------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_repository_path = create_experiment_directory(
        results_dir=RESULTS_DIR,
        experiment_type=experiment_type,
        dataset_name=dataset_name,
        timestamp=timestamp,
        experiment_tag=experiment_tag,
    )
    # networks are loaded from this object
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=MODELS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_list = sdp_crown_models_loading(MODELS_DIR, input_shape, device)
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
    # Apply SDP-CROWN preprocessing for CIFAR-10 images (normalization with SDP-CROWN means/std)
    # Note: flatten=True because images are flattened when passed to attacks/verifiers
    # Preprocessing is applied BEFORE flattening (transform order: ToTensor -> Preprocess -> Flatten)
    if sample_stratified:
        cifar10_torch_dataset, original_indices = get_balanced_sample(
            dataset_name=dataset_name,
            train_bool=(split == "train"),
            dataset_size=sample_size,
            dataset_dir=DATASET_DIR,
            seed=random_seed,
            flatten=True,
            sdpcrown_preprocess=True,
        )
    else:
        cifar10_torch_dataset, original_indices = get_sample(
            dataset_name=dataset_name,
            train_bool=(split == "train"),
            dataset_size=sample_size,
            dataset_dir=DATASET_DIR,
            seed=random_seed,
            flatten=True,
            sdpcrown_preprocess=True,
        )
    # Pass original_indices to maintain mapping to original dataset
    dataset = PytorchExperimentDataset(dataset=cifar10_torch_dataset, original_indices=original_indices.tolist())

    # ----------------------------------------VERIFICATION CONFIGURATION---------------------------------------------
    property_generator = One2AnyPropertyGenerator()

    pgd_iterations = 100
    pgd_rel_stepsize = 0.03
    pgd_random_start = True
    pgd_restarts = 15
    # SDP-CROWN CIFAR-10 preprocessing normalizes by std=0.225 (see `SDPCrownCIFAR10Preprocess`)
    # we pass `std_rescale_factor=0.225` to normalize the epsilon, so epsilons remain in pixel space
    # and are converted inside the attack.
    std_rescale_factor = 0.225

    attack_configs = [
        # {
        #     "name": "pgd_l2_single",
        #     "attack": PGDAttack(
        #         number_iterations=pgd_iterations,
        #         rel_stepsize=pgd_rel_stepsize,
        #         randomise=pgd_random_start,
        #         norm="l2",
        #         bounds=None,
        #         std_rescale_factor=std_rescale_factor,
        #     ),
        #     "attack_type": "PGD",
        #     "attack_iterations": pgd_iterations,
        # },
        {
            "name": "pgd_l2",
            "attack": RestartingPGDAttack(
                number_iterations=pgd_iterations,
                n_restarts=pgd_restarts,
                rel_stepsize=pgd_rel_stepsize,
                randomise=pgd_random_start,
                norm="l2",
                std_rescale_factor=std_rescale_factor,
                top_k=1,
                early_stop_on_success=True,
            ),
            "attack_type": "PGD",
            "attack_iterations": pgd_iterations,
        },
    ]

    first_attack_name = attack_configs[0]["name"]
    experiment_repository.initialize_new_experiment(first_attack_name)

    # ----------------------------------------SAVE ORIGINAL DATASET INDICES----------------------------------------
    indices_file = save_original_indices(
        dataset_name=dataset_name,
        original_indices=original_indices,
        output_dir=experiment_repository.get_act_experiment_path(),
        sample_size=sample_size,
        split=split,
    )
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
                experiment_tag=experiment_tag,
            )

            # Log attack-specific parameters
            attack_params = {
                **base_params,
                "experiment_name": attack_name,
                "attack_type": attack_config["attack_type"],
                "attack_iterations": attack_config["attack_iterations"],
            }
            comet_tracker.log_parameters(attack_params)

            # Log classifier metrics for this attack experiment
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
