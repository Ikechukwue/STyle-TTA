from pathlib import Path
import os
from .constants import PRETRAINED_MODELS
# ============================================================================
# Project paths
# ============================================================================

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent

# ============================================================================
# Local paths
# ============================================================================

DATA_PATH = Path(
    os.getenv("DATA_PATH", "/data/local/retristyle/data")
)

OUTPUT_PATH = Path(
    os.getenv("OUTPUT_PATH", PROJECT_ROOT / "results")
)

WEIGHTS_DIR = Path(
    os.getenv(
        "WEIGHTS_DIR",
        "/data/local/retristyle/models/style_transfer",
    )
)

EMBEDDING_DIR = Path(
    os.getenv(
        "EMBEDDING_DIR",
        PROJECT_ROOT / "data" / "embeddings",
    )
)

MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        PROJECT_ROOT / "data" / "models",
    )
)

# ============================================================================
# YAML configs
# ============================================================================

GPU_CONFIG_DIR = CONFIG_DIR / "gpu"


def gpu_config(n_gpus: int) -> Path:
    return GPU_CONFIG_DIR / f"gpu_{n_gpus:02d}.yaml"
