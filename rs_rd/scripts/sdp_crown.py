import logging
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from autoverify.verifier import SDPCrown

try:
    try:
        sdp_crown_tool_dir = SDPCrown().tool_path
    except Exception:
        sdp_crown_tool_dir = Path.home() / ".local/share/autoverify/verifiers/sdpcrown/tool"
    sys.path.insert(0, str(sdp_crown_tool_dir))
    from models import (
        CIFAR10_CNN_A,
        CIFAR10_CNN_B,
        CIFAR10_CNN_C,
        MNIST_MLP,
        CIFAR10_ConvDeep,
        CIFAR10_ConvLarge,
        CIFAR10_ConvSmall,
        MNIST_ConvLarge,
        MNIST_ConvSmall,
    )

    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    logging.warning(
        "Could not import SDP-CROWN models from %s. State_dict loading will not work. Error: %s",
        sdp_crown_tool_dir,
        e,
    )

import ada_verona.util.logger as logger
from ada_verona import (
    AutoVerifyModule,
    BinarySearchEpsilonValueEstimator,
    CometTracker,
    ExperimentRepository,
    IdentitySampler,
    One2AnyPropertyGenerator,
    PredictionsBasedSampler,
    PytorchExperimentDataset,
    create_distribution,
    get_balanced_sample,
    get_dataset_dir,
    get_models_dir,
    get_results_dir,
    get_sample,
    log_classifier_metrics,
    log_verona_experiment_summary,
    log_verona_results,
)
from ada_verona.database.machine_learning_model.pytorch_network import PyTorchNetwork

logger.setup_logging(level=logging.INFO)


logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("dulwich").setLevel(logging.WARNING)
logging.getLogger("comet_ml").setLevel(logging.INFO)


def infer_model_architecture(model_name: str) -> nn.Module:
    """
    Infer PyTorch model architecture from model filename.

    Args:
        model_name: Model filename (without extension)

    Returns:
        Instantiated model architecture
    """
    if not MODELS_AVAILABLE:
        raise ImportError(
            "SDP-CROWN models are not available. Cannot load state_dict files. "
            "Please ensure SDP-CROWN is available in the expected location."
        )

    name = model_name.lower()

    if "mnist" in name:
        if "mlp" in name:
            return MNIST_MLP()
        elif "convsmall" in name:
            return MNIST_ConvSmall()
        else:
            return MNIST_ConvLarge()

    elif "cifar" in name or "cifar10" in name:
        if "cnn_a" in name:
            return CIFAR10_CNN_A()
        elif "cnn_b" in name:
            return CIFAR10_CNN_B()
        elif "cnn_c" in name:
            return CIFAR10_CNN_C()
        elif "convsmall" in name:
            return CIFAR10_ConvSmall()
        elif "convdeep" in name:
            return CIFAR10_ConvDeep()
        elif "convlarge" in name or "conv_large" in name:
            return CIFAR10_ConvLarge()
        else:
            # Default to ConvLarge for CIFAR-10
            return CIFAR10_ConvLarge()

    else:
        raise ValueError(
            f"Could not infer architecture from model name '{model_name}'. "
            "Please use one of the known SDP-CROWN architectures."
        )


def load_pytorch_model(model_path: Path, device: torch.device) -> nn.Module:
    """
    Load a PyTorch model from a .pth file, handling both state_dict and full model objects.

    Args:
        model_path: Path to the .pth file
        device: PyTorch device to load the model on

    Returns:
        Loaded PyTorch model
    """
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    # Check if loaded object is a state_dict (OrderedDict or dict)
    if isinstance(loaded, (OrderedDict, dict)) and not isinstance(loaded, nn.Module):
        # state_dict, need to instantiate model architecture first
        logging.info(f"Detected state_dict in {model_path.name}, inferring architecture from filename")
        model = infer_model_architecture(model_path.stem)
        model.load_state_dict(loaded)
        model = model.to(device)
        model.eval()
        return model
    elif isinstance(loaded, nn.Module):
        model = loaded.to(device)
        model.eval()
        return model
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {model_path}. "
            "Expected either a state_dict (OrderedDict/dict) or a full model (nn.Module)."
        )


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Log experiment start time
    start_time = time.time()

    # ---------------------------------------Basic Experiment Settings -----------------------------------------
    dataset_name = "CIFAR-10"
    input_shape = (1, 3, 32, 32)
    split = "test"
    sample_size = 300
    random_seed = 5432

    use_identity_sampler = False
    sample_correct_predictions = True
    sample_stratified = False

    experiment_name = "sdpcrown_300"
    experiment_tag = experiment_name

    # ----------------------------------------VERIFIER CONFIGURATION------------------------------------------
    config_path = Path(__file__).parent.parent / "config" / "SDP-crown-conf.yaml"
    timeout = 300

    # ----------------------------------------PERTURBATION CONFIGURATION------------------------------------------
    epsilon_start = 0.00
    epsilon_stop = 0.3
    epsilon_step = 2 / 255
    epsilon_list = np.arange(epsilon_start, epsilon_stop, epsilon_step)
    # ----------------------------------------DATASET AND MODELS DIRECTORY CONFIGURATION---------------------------
    DATASET_DIR = get_dataset_dir(dataset_name)
    MODELS_DIR = get_models_dir(dataset_name) / experiment_name
    RESULTS_DIR = get_results_dir()

    # --------- -----------------------------COMET ML TRACKING INITIALIZATION --------------------------------
    comet_tracker = CometTracker(project_name="rs-rd", auto_login=True)

    # ----------------------------------------EXPERIMENT REPOSITORY CONFIGURATION----------------------------------
    experiment_dir_name = (
        f"verona_rs_rd_{experiment_name}_{dataset_name}_{sample_size}_"
        f"sample_correct_{sample_correct_predictions}_"
        f"sample_stratified_{sample_stratified}"
    )
    experiment_repository_path = Path(RESULTS_DIR) / experiment_dir_name
    os.makedirs(experiment_repository_path, exist_ok=True)
    # networks are loaded from this object
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=MODELS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models from {MODELS_DIR}")
    network_list = []
    for model_path in MODELS_DIR.glob("*.pth"):
        try:
            model = load_pytorch_model(model_path, device)
            network_list.append(
                PyTorchNetwork(
                    model=model,
                    input_shape=input_shape,
                    name=model_path.stem,
                    path=model_path,
                )
            )
            logging.info(f"Successfully loaded model: {model_path.name}")
        except Exception as e:
            logging.error(f"Failed to load model {model_path.name}: {e}", exc_info=True)

    model_names = [network.name for network in network_list]

    epsilon_tag = f"eps_{epsilon_start}_{epsilon_stop}_{epsilon_step}"
    # Comet experiment start
    comet_tracker.start_experiment(
        experiment_name=f"rs_rd_{experiment_name}_{dataset_name}_{timestamp}",
        tags=[experiment_name, dataset_name, epsilon_tag, *model_names],
        experiment_tag=experiment_tag,
    )
    experiment_repository.initialize_new_experiment(experiment_name)

    # Log experiment parameters to Comet ML
    comet_tracker.log_parameters(
        {
            "dataset_name": dataset_name,
            "split": split,
            "sample_size": sample_size,
            "sample_correct_predictions": sample_correct_predictions,
            "experiment_name": experiment_name,
            "epsilon_start": epsilon_start,
            "epsilon_stop": epsilon_stop,
            "epsilon_step": epsilon_step,
            "sampling_method": "StratifiedShuffleSplit",
            "random_seed_dataset_sampling": random_seed,
        }
    )

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
                zipped_preds = zip(metrics["labels"], metrics["predictions"], strict=True)
                for i, (true_label, pred_label) in enumerate(zipped_preds):
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
    robustness_attack_estimator = AutoVerifyModule(verifier=SDPCrown(), timeout=timeout, config=Path(config_path))

    # ----------------------------------------EPSILON VALUE SEARCH CONFIGURATION-------------------------------------
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=robustness_attack_estimator
    )

    # ----------------------------------------CREATE ROBUSTNESS DISTRIBUTION------------------------------------------
    logging.info("=" * 80)
    logging.info("Starting robustness verification (create_distribution)...")
    logging.info(f"Will verify {len(network_list)} network(s)")
    logging.info(f"Dataset size: {len(dataset)} samples")
    logging.info(f"Epsilon range: {epsilon_start} to {epsilon_stop} (step: {epsilon_step}, {len(epsilon_list)} values)")
    logging.info(f"Timeout per verification: {timeout} seconds")

    # Log environment info for debugging subprocess issues
    logging.info("Environment check for subprocess execution:")
    logging.info(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not set')}")
    logging.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logging.info(f"  CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")

    try:
        logging.info("Starting robustness verification...")
        create_distribution(
            experiment_repository,
            dataset,
            dataset_sampler,
            epsilon_value_estimator,
            property_generator,
            network_list=network_list,
        )
        logging.info("Robustness verification completed successfully")
    except KeyboardInterrupt:
        logging.warning("Verification interrupted by user")
        raise
    except MemoryError as e:
        logging.error(f"Out of Memory error during verification: {e}")
        raise
    except subprocess.SubprocessError as e:
        logging.error(f"Subprocess error during verification: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during robustness verification: {e}", exc_info=True)
        raise
    results_path = experiment_repository.get_results_path()

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
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user")
        raise
    except Exception as e:
        logging.critical(f"Uncaught exception in main: {e}", exc_info=True)
        raise
