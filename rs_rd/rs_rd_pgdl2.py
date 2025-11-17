import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# experiment utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from comet_tracker import CometTracker, log_classifier_metrics, log_verona_experiment_summary, log_verona_results
from experiment_utils import compute_classifier_metrics, create_distribution, get_balanced_sample
from rs_rd_research.paths import get_dataset_dir, get_models_dir, get_results_dir

import ada_verona.util.logger as logger
from ada_verona import (
    AttackEstimationModule,
    BinarySearchEpsilonValueEstimator,
    ExperimentRepository,
    One2AnyPropertyGenerator,
    PGDAttack,
    PredictionsBasedSampler,
    IdentitySampler,
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
    sample_size = 10
    random_seed = 42

    use_identity_sampler = True  
    sample_correct_predictions = False
    
    
    experiment_type = "verona_upper_bounding"
    experiment_name = "pgd_l2"

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 0.4
    epsilon_step = 0.0039 # why ~1/8 of 8/255 epsilon step size?
    # ----------------------------------------DATASET AND MODELS DIRECTORY CONFIGURATION---------------------------
    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name) / experiment_type
    RESULTS_DIR = get_results_dir()

    # --------- -----------------------------COMET ML TRACKING INITIALIZATION --------------------------------
    comet_tracker = CometTracker(project_name="rs-rd", auto_login=True)
    
    # ----------------------------------------EXPERIMENT REPOSITORY CONFIGURATION----------------------------------
    experiment_repository_path = (
        Path(RESULTS_DIR)
        / f"verona_rs_rd_{experiment_name}_{dataset_name}_{sample_size}_sample_correct_{sample_correct_predictions}"
    )
    os.makedirs(experiment_repository_path, exist_ok=True)
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=MODELS_DIR)
    
    network_list = experiment_repository.get_network_list()
    model_names = [network.name for network in network_list]
    
    epsilon_tag = f"eps_{epsilon_start}_{epsilon_stop}_{epsilon_step}"
    #Comet experiment start
    comet_tracker.start_experiment(
        experiment_name=f"rs_rd_pgd_l2_CIFAR-10_{timestamp}",
        tags=[experiment_type, dataset_name, experiment_name, epsilon_tag, *model_names],
    )
    experiment_repository.initialize_new_experiment(experiment_name)

    # Log experiment parameters to Comet ML
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
            "attack_iterations": 40,   #TODO make this dynamic
            "epsilon_start": epsilon_start,
            "epsilon_stop": epsilon_stop,
            "epsilon_step": epsilon_step,
            "sampling_method": "StratifiedShuffleSplit",
            "random_seed_dataset_sampling": random_seed,
        }
    )

    # ----------------------------------------DATASET CONFIGURATION-----------------------------------------------
    # load dataset and get original CIFAR-10 indices
    cifar10_torch_dataset, original_indices = get_balanced_sample(
        dataset_name=dataset_name,
        train_bool=(split == "train"),
        dataset_size=sample_size,
        dataset_dir=DATASET_DIR,
        seed=random_seed,
    )
    # Pass original_indices to maintain mapping to original dataset
    dataset = PytorchExperimentDataset(dataset=cifar10_torch_dataset, original_indices=original_indices.tolist())

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

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 0.4
    epsilon_step = 0.0039
    epsilon_list = np.arange(epsilon_start, epsilon_stop, epsilon_step)

    # ----------------------------------------DATASET SAMPLER CONFIGURATION------------------------------------------
    #If want full dataset, use IdentitySampler
   
    if  use_identity_sampler:
        dataset_sampler = IdentitySampler()
    else:
        dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=sample_correct_predictions)
    # ----------------------------------------CLASSIFIER PERFORMANCE METRICS-----------------------------------------
    # Compute and log classifier metrics for each network before running robustness verification
    # TODO: currently not the most efficient, as we do inference twice 
    logging.info(f"Computing classifier metrics for {len(network_list)} network(s)")

    for network in network_list:
        try:
            # Compute metrics on full dataset (before sampling) #TODO: kinda pointless for PredictionsBasedSampler
            metrics = dataset_sampler.compute_metrics(network, dataset)

            # Log metrics to Comet ML
            log_classifier_metrics(comet_tracker, network.name, metrics)

            # Log detailed predictions to file
            predictions_file = (
                experiment_repository.get_act_experiment_path() / f"{network.name}_predictions_{timestamp}.txt"
            )
            with open(predictions_file, "w") as f:
                f.write(f"Network: {network.name}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Correct: {metrics['correct']}/{metrics['total']}\n\n")
                f.write("Sample_ID,True_Label,Predicted_Label,Correct\n")
                for i, (true_label, pred_label) in enumerate(zip(metrics["labels"], metrics["predictions"])):
                    is_correct = "Yes" if true_label == pred_label else "No"
                    f.write(f"{i},{true_label},{pred_label},{is_correct}\n")

            logging.info(f"Saved metrics to {predictions_file}")

            # Log predictions file to Comet ML
            comet_tracker.log_asset(predictions_file)

        except Exception as e:
            logging.error(f"Failed to compute metrics for network {network.name}: {e}")

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
    results_path = experiment_repository.get_results_path()
  
    # Log result files (image_id column now contains original CIFAR-10 indices)
    log_verona_results(comet_tracker, results_path)
    logging.info("Result files contain original dataset indices in the image_id column")
    
    # ----------------------------------------LOG RESULTS TO COMET ML---------------------------------------------
    experiment_path = experiment_repository.get_act_experiment_path()

    # Log experiment summary
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
            "type": f"{experiment_name}",
            "iterations": 40,
        },
        epsilon_info={
            "start": epsilon_start,
            "stop": epsilon_stop,
            "step": epsilon_step,
            "list": epsilon_list,
        },
    )

    # Log final metrics
    comet_tracker.log_metrics({"total_duration_seconds": time.time() - start_time})

    # End the experiment
    comet_tracker.end_experiment()


if __name__ == "__main__":
    main()
