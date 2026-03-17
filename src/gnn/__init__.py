import os

# Allow selecting model via environment variable, fallback to model03
_model_choice = os.getenv("SHOOTER_MODEL", "model03")

from src.gnn.common import (
    CallbackManager, build_opt, get_acc, get_acc_random,
    log, ensure_tensor, make_weighted_loss
    )

if _model_choice == "model03":
    from .model03.model import train, validate, test, create_model, get_info, neighbor_probs
else:
    raise ImportError(f"Unknown model choice: {_model_choice}")

__all__ = ["train", "validate", "test", "create_model", "get_info", "neighbor_probs"]
