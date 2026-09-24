"""Persistence utilities for saving/loading models, preprocessors, and scorers."""
import os
import pickle
import torch

def save_pipeline(save_dir: str, model=None, preprocessor=None, scorer=None, contract=None):
    """Save all pipeline components to a directory."""
    os.makedirs(save_dir, exist_ok=True)

    if model is not None:
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": model.encoder[0].in_features,
            "emb_dim": model.encoder[0].out_features,
            "head_dim": model.head[0].out_features,
        }, os.path.join(save_dir, "scarf_model.pt"))

    if preprocessor is not None:
        with open(os.path.join(save_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(preprocessor, f)

    if scorer is not None:
        with open(os.path.join(save_dir, "scorer.pkl"), "wb") as f:
            pickle.dump(scorer, f)

    if contract is not None:
        with open(os.path.join(save_dir, "contract.pkl"), "wb") as f:
            pickle.dump(contract, f)

def load_pipeline(save_dir: str, model_class=None, input_dim: int = None):
    """Load all pipeline components from a directory."""
    result = {}

    model_path = os.path.join(save_dir, "scarf_model.pt")
    if os.path.exists(model_path) and model_class is not None:
        checkpoint = torch.load(model_path, weights_only=False)
        model = model_class(
            input_dim=checkpoint["input_dim"],
            emb_dim=checkpoint["emb_dim"],
            head_dim=checkpoint["head_dim"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        result["model"] = model

    prep_path = os.path.join(save_dir, "preprocessor.pkl")
    if os.path.exists(prep_path):
        with open(prep_path, "rb") as f:
            result["preprocessor"] = pickle.load(f)

    scorer_path = os.path.join(save_dir, "scorer.pkl")
    if os.path.exists(scorer_path):
        with open(scorer_path, "rb") as f:
            result["scorer"] = pickle.load(f)

    contract_path = os.path.join(save_dir, "contract.pkl")
    if os.path.exists(contract_path):
        with open(contract_path, "rb") as f:
            result["contract"] = pickle.load(f)

    return result
