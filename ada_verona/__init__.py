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

"""
ADA-VERONA: Neural Network Robustness Analysis Framework

A framework for analyzing neural network robustness
through verification and adversarial testing.
"""

import importlib.util
import warnings
from importlib.metadata import version

# Database classes
from .database.dataset.data_point import DataPoint
from .database.dataset.experiment_dataset import ExperimentDataset
from .database.dataset.image_file_dataset import ImageFileDataset
from .database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from .database.epsilon_status import EpsilonStatus
from .database.epsilon_value_result import EpsilonValueResult
from .database.experiment_repository import ExperimentRepository
from .database.machine_learning_model.network import Network
from .database.machine_learning_model.onnx_network import ONNXNetwork
from .database.machine_learning_model.pytorch_network import PyTorchNetwork
from .database.machine_learning_model.torch_model_wrapper import TorchModelWrapper
from .database.verification_context import VerificationContext
from .database.verification_result import VerificationResult
from .database.vnnlib_property import VNNLibProperty

# Dataset sampler classes
from .dataset_sampler.dataset_sampler import DatasetSampler
from .dataset_sampler.identity_sampler import IdentitySampler
from .dataset_sampler.predictions_based_sampler import PredictionsBasedSampler

# Epsilon value estimator classes
from .epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from .epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from .epsilon_value_estimator.iterative_epsilon_value_estimator import (
    IterativeEpsilonValueEstimator,
)

# Verification module classes
from .verification_module.attack_estimation_module import AttackEstimationModule
from .verification_module.attacks.attack import Attack
from .verification_module.attacks.fgsm_attack import FGSMAttack
from .verification_module.attacks.pgd_attack import PGDAttack
from .verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from .verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)
from .verification_module.property_generator.property_generator import PropertyGenerator
from .verification_module.verification_module import VerificationModule

__version__ = version("ada-verona")
__author__ = "ADA Research Group"
# Check for autoattack availability
HAS_AUTOATTACK = importlib.util.find_spec("autoattack") is not None
if not HAS_AUTOATTACK:
    warnings.warn(
        "AutoAttack not found. Some adversarial attack features will be limited. "
        "To install: uv pip install git+https://github.com/fra31/auto-attack",
        stacklevel=2,
    )

# Check for autoverify availability
HAS_AUTOVERIFY = importlib.util.find_spec("autoverify") is not None
if not HAS_AUTOVERIFY:
    warnings.warn(
        "AutoVerify not found. Some complete verification features will be limited. "
        "To install: uv pip install auto-verify>=0.1.4",
        stacklevel=2,
    )

HAS_FOOLBOX = importlib.util.find_spec("foolbox") is not None
if not HAS_FOOLBOX:
    warnings.warn(
        "Foolbox not found. Some adversarial attack features will be limited. To install: uv pip install foolbox",
        stacklevel=2,
    )

# Import rs_rd utilities if available
try:
    import sys
    from pathlib import Path

    # Add parent directory to path to allow importing rs_rd
    _ada_verona_path = Path(__file__).parent.parent
    if str(_ada_verona_path) not in sys.path:
        sys.path.insert(0, str(_ada_verona_path))

    # Import rs_rd utilities
    from rs_rd.utils import comet_tracker, experiment_utils, paths

    CometTracker = comet_tracker.CometTracker
    log_classifier_metrics = comet_tracker.log_classifier_metrics
    log_verona_experiment_summary = comet_tracker.log_verona_experiment_summary
    log_verona_results = comet_tracker.log_verona_results

    create_distribution = experiment_utils.create_distribution
    create_experiment_directory = experiment_utils.create_experiment_directory
    get_balanced_sample = experiment_utils.get_balanced_sample
    get_sample = experiment_utils.get_sample
    get_dataset_config = experiment_utils.get_dataset_config
    save_original_indices = experiment_utils.save_original_indices
    load_networks_from_directory = experiment_utils.load_networks_from_directory
    sdp_crown_models_loading = experiment_utils.sdp_crown_models_loading
    load_sdpcrown_pytorch_model = experiment_utils.load_sdpcrown_pytorch_model

    get_dataset_dir = paths.get_dataset_dir
    get_models_dir = paths.get_models_dir
    get_results_dir = paths.get_results_dir

    HAS_RS_RD_UTILS = True
except (ImportError, AttributeError):
    HAS_RS_RD_UTILS = False
    CometTracker = None
    log_classifier_metrics = None
    log_verona_experiment_summary = None
    log_verona_results = None
    create_distribution = None
    create_experiment_directory = None
    get_balanced_sample = None
    get_sample = None
    get_dataset_config = None
    save_original_indices = None
    get_dataset_dir = None
    get_models_dir = None
    get_results_dir = None
    load_networks_from_directory = None
    sdp_crown_models_loading = None
    load_sdpcrown_pytorch_model = None


__all__ = [
    "__version__",
    "__author__",
    "HAS_AUTOATTACK",
    "HAS_AUTOVERIFY",
    "HAS_FOOLBOX",
    "HAS_RS_RD_UTILS",
    # Core abstract classes
    "DatasetSampler",
    "EpsilonValueEstimator",
    "VerificationModule",
    "Network",
    "PropertyGenerator",
    "Attack",
    "ExperimentDataset",
    # Database classes
    "ExperimentRepository",
    "VerificationContext",
    "ONNXNetwork",
    "PyTorchNetwork",
    "TorchModelWrapper",
    "VNNLibProperty",
    "VerificationResult",
    "EpsilonValueResult",
    "EpsilonStatus",
    "DataPoint",
    # Dataset sampler classes
    "PredictionsBasedSampler",
    "IdentitySampler",
    "PytorchExperimentDataset",
    "ImageFileDataset",
    # Epsilon value estimator classes
    "BinarySearchEpsilonValueEstimator",
    "IterativeEpsilonValueEstimator",
    # Verification module classes
    "AttackEstimationModule",
    "PGDAttack",
    "FGSMAttack",
    # Property generator classes
    "One2AnyPropertyGenerator",
    "One2OnePropertyGenerator",
    # rs_rd utility classes and functions
    "CometTracker",
    "log_classifier_metrics",
    "log_verona_experiment_summary",
    "log_verona_results",
    "create_distribution",
    "create_experiment_directory",
    "get_balanced_sample",
    "get_sample",
    "get_dataset_config",
    "save_original_indices",
    "get_dataset_dir",
    "get_models_dir",
    "get_results_dir",
    "load_networks_from_directory",
    "sdp_crown_models_loading",
    "load_sdpcrown_pytorch_model",
]


if HAS_AUTOATTACK:
    auto_attack_module = importlib.import_module(".verification_module.attacks.auto_attack_wrapper", __package__)
    AutoAttackWrapper = auto_attack_module.AutoAttackWrapper
    __all__.extend(["AutoAttackWrapper"])

if HAS_AUTOVERIFY:
    autoverify_module = importlib.import_module(".verification_module.auto_verify_module", __package__)
    AutoVerifyModule = autoverify_module.AutoVerifyModule
    parse_counter_example = autoverify_module.parse_counter_example
    parse_counter_example_label = autoverify_module.parse_counter_example_label
    __all__.extend(
        [
            "AutoVerifyModule",
            "parse_counter_example",
            "parse_counter_example_label",
        ]
    )

if HAS_FOOLBOX:
    foolbox_module = importlib.import_module(".verification_module.attacks.foolbox_attack", __package__)
    FoolboxAttack = foolbox_module.FoolboxAttack
    __all__.extend(["FoolboxAttack"])
