from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint
from .config import DEFAULT_CONFIG
from .data import KGSSGNNDataset, collate_kgssgnn, load_local_drugbank, precompute_features
from .models import KGSSGNN
from .utils import get_device


def evaluate_model(model: KGSSGNN, loader: DataLoader, n_classes: int, device: torch.device) -> dict:
    model.eval()
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for graph_a, graph_b, fp_a, fp_b, kg_a, kg_b, target in loader:
            logits = model(
                graph_a.to(device),
                graph_b.to(device),
                fp_a.to(device),
                fp_b.to(device),
                kg_a.to(device),
                kg_b.to(device),
            )
            batch_probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs.append(batch_probs)
            preds.extend(batch_probs.argmax(axis=1).tolist())
            labels.extend(target.numpy().tolist())

    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    y_prob = np.vstack(probs)
    auroc = auprc = float("nan")
    try:
        present = np.unique(y_true)
        if len(present) >= 2:
            y_prob_present = y_prob[:, present]
            y_prob_present = y_prob_present / y_prob_present.sum(axis=1, keepdims=True)
            remap = {label: idx for idx, label in enumerate(present)}
            y_true_remap = np.asarray([remap[label] for label in y_true])
            auroc = roc_auc_score(y_true_remap, y_prob_present, multi_class="ovr", average="macro")
            auprc = float(
                np.mean(
                    [
                        average_precision_score((y_true_remap == idx).astype(int), y_prob_present[:, idx])
                        for idx in range(len(present))
                    ]
                )
            )
    except Exception:
        pass

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(auroc),
        "auprc": float(auprc),
    }


def evaluate_checkpoint(checkpoint: str, data_path: str, batch_size: int) -> dict:
    device = get_device()
    payload = load_checkpoint(checkpoint, device)
    bundle = load_local_drugbank(data_path)
    graph_cache, fp_cache, kg_cache = precompute_features(bundle.smiles_map)
    test_set = KGSSGNNDataset(bundle.test, graph_cache, fp_cache, kg_cache)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_kgssgnn)
    model = KGSSGNN(n_classes=bundle.n_classes).to(device)
    model.load_state_dict(payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload)
    return evaluate_model(model, test_loader, bundle.n_classes, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a KG-SS-GNN checkpoint.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CONFIG.checkpoint_path))
    parser.add_argument("--data-path", default=str(DEFAULT_CONFIG.data_path))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(args.checkpoint, args.data_path, args.batch_size)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
