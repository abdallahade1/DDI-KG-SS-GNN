from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .checkpoints import load_checkpoint
from .config import DEFAULT_CONFIG
from .features import fingerprint, kg_features, molecular_descriptors, smiles_to_graph
from .models import KGSSGNN
from .utils import get_device


class KGSSGNNPredictor:
    def __init__(self, checkpoint: str | Path = DEFAULT_CONFIG.checkpoint_path):
        self.device = get_device()
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"KG-SS-GNN checkpoint not found at {self.checkpoint.resolve()}. "
                "Expected the notebook checkpoint under project/checkpoint."
            )

        payload = load_checkpoint(self.checkpoint, self.device)
        if isinstance(payload, dict) and "model_state" in payload:
            state_dict = payload["model_state"]
            self.labels = [str(label) for label in payload["labels"]]
            n_classes = int(payload["n_classes"])
        else:
            state_dict = payload
            n_classes = int(state_dict.get("classifier.8.bias").shape[0])
            self.labels = [f"Interaction type {i}" for i in range(n_classes)]

        self.model = KGSSGNN(n_classes=n_classes).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, smiles_a: str, smiles_b: str, top_k: int = 5) -> dict:
        graph_a = smiles_to_graph(smiles_a)
        graph_b = smiles_to_graph(smiles_b)
        fp_a = fingerprint(smiles_a).unsqueeze(0).to(self.device)
        fp_b = fingerprint(smiles_b).unsqueeze(0).to(self.device)
        kg_a = kg_features(smiles_a).unsqueeze(0).to(self.device)
        kg_b = kg_features(smiles_b).unsqueeze(0).to(self.device)
        batch_a = Batch.from_data_list([graph_a]).to(self.device)
        batch_b = Batch.from_data_list([graph_b]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_a, batch_b, fp_a, fp_b, kg_a, kg_b)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        order = np.argsort(probs)[::-1][:top_k]
        return {
            "descriptors_a": molecular_descriptors(smiles_a),
            "descriptors_b": molecular_descriptors(smiles_b),
            "predictions": [
                {"label": self.labels[i], "probability": float(probs[i])}
                for i in order
            ],
        }
