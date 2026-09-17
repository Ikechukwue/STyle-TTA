"""
experiments.tta — Test-Time Adaptation Inference Package
=========================================================

This package splits the TTA inference pipeline into focused modules:

- :mod:`constants`           — Shared constants, defaults, enumerations
- :mod:`augmentation`        — View generation (geometric, style-transfer, distributed)
- :mod:`evaluation`          — Evaluation strategies (vanilla, ZERO, FOODS)
- :mod:`checkpoint`          — Prediction persistence and resume logic
- :mod:`reference_db_setup`  — Reference database & retriever construction
- :mod:`extract_embeddings`  — Standalone embedding extraction & caching
- :mod:`run_inference`       — Main inference loop and CLI entry-point

Quick-start
-----------
::

    # Single-GPU, 1 reference (local testing)
    python -m experiments.tta.run_inference \\
        --dataset pathmnist --data_path ./data \\
        --classifier densenet121 --weights_path ./checkpoints/model.pth \\
        --tta_method style_tta --eval_strategy zero \\
        --retrieval_strategy random --n_refs 1

    # Extract embeddings independently
    python -m experiments.tta.extract_embeddings \\
        --dataset pathmnist --data_path ./data \\
        --output_dir ./embeddings --splits train test

    # Evaluate stored predictions
    python -m experiments.evaluate_predictions \\
        --predictions_path ./results/tta_inference/predictions/epistr.json
"""

from .constants import (
    DEFAULT_SEED,
    ALL_SEEDS,
    ZERO_N_VIEWS,
    ZERO_GAMMA,
    DEFAULT_N_VIEWS,
    AUGMENTATION_TTA_METHODS,
    RETRIEVAL_TTA_METHODS,
    AVAILABLE_TTA_METHODS,
    AVAILABLE_EVAL_STRATEGIES,
    AVAILABLE_RETRIEVAL_STRATEGIES,
    TRAINING_AUGMENTATIONS,
)
from .augmentation import augment_views, augment_views_distributed
from .evaluation import eval_vanilla, eval_zero, eval_foods, eval_tent
from .checkpoint import (
    build_experiment_key,
    build_experiment_tag,
    predictions_path,
    results_path,
    load_predictions,
    save_predictions,
    load_results,
    save_result,
)
from .reference_db_setup import build_reference_db, build_retriever, materialise_images

# extract_embeddings and run_inference are standalone CLI entry-points.
# They are NOT eagerly imported here to avoid the "found in sys.modules
# after import of package … but prior to execution" RuntimeWarning that
# Python emits when running  `python -m experiments.tta.extract_embeddings`
# (or run_inference).  Import them explicitly where needed:
#
#   from code.experiments.tta.extract_embeddings import extract_and_cache, ...
#   from code.experiments.tta.run_inference import run_inference, build_parser
