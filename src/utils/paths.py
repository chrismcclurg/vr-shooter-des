from pathlib import Path

# --------------------------------------------------
# Project root (assumes this file is in src/utils/)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------
# Top-level directories
# --------------------------------------------------
SRC_DIR   = PROJECT_ROOT / "src"
DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------
# Data subdirectories
# --------------------------------------------------
ENV_DIR        = DATA_DIR / "environments"
CACHE_DIR      = DATA_DIR / "cache"
RESULTS_DIR    = DATA_DIR / "results"
PARTICIPANT_DIR = DATA_DIR / "participants"
SHOOTER_DIR    = DATA_DIR / "shooters"
VISUAL_DIR     = DATA_DIR / "visual"

# --------------------------------------------------
# Model subdirectories
# --------------------------------------------------
RL_MODEL_DIR = MODEL_DIR / "rl"
GNN_MODEL_DIR = MODEL_DIR / "gnn"

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def get_env_file(name: str) -> Path:
    """Return path to environment file by name."""
    return ENV_DIR / name

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path