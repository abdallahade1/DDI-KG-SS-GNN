from __future__ import annotations

import argparse
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import DEFAULT_CONFIG
from .data import KGSSGNNDataset, collate_kgssgnn, load_local_drugbank, precompute_features, weighted_sampler
from .evaluate import evaluate_model
from .models import FocalLoss, KGSSGNN
from .utils import ensure_dir, get_device, set_seed


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    bundle = load_local_drugbank(args.data_path, seed=args.seed)
    graph_cache, fp_cache, kg_cache = precompute_features(bundle.smiles_map)

    train_set = KGSSGNNDataset(bundle.train, graph_cache, fp_cache, kg_cache)
    valid_set = KGSSGNNDataset(bundle.valid, graph_cache, fp_cache, kg_cache)
    test_set = KGSSGNNDataset(bundle.test, graph_cache, fp_cache, kg_cache)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=weighted_sampler(bundle.train["label"].to_numpy(), bundle.n_classes),
        collate_fn=collate_kgssgnn,
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_kgssgnn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_kgssgnn)

    model = KGSSGNN(n_classes=bundle.n_classes).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            graph_a, graph_b, fp_a, fp_b, kg_a, kg_b, target = batch
            optimizer.zero_grad()
            logits = model(
                graph_a.to(device),
                graph_b.to(device),
                fp_a.to(device),
                fp_b.to(device),
                kg_a.to(device),
                kg_b.to(device),
            )
            loss = criterion(logits, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch == 1 or epoch % args.eval_every == 0:
            val_metrics = evaluate_model(model, valid_loader, bundle.n_classes, device)
            print(
                f"epoch={epoch:03d} loss={total_loss / max(1, len(train_loader)):.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_auroc={val_metrics['auroc']:.4f} "
                f"val_recall={val_metrics['recall']:.4f}"
            )
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, bundle.n_classes, device)
    print(f"training_seconds={time.time() - start:.0f}")
    print("test_metrics=" + str({k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)}))

    output_dir = ensure_dir(args.output_dir)
    checkpoint_path = output_dir / "kgssgnn_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": [str(label) for label in bundle.label_encoder.classes_],
            "n_classes": bundle.n_classes,
            "metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"saved_checkpoint={checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train KG-SS-GNN on local DrugBank DDI CSV.")
    parser.add_argument("--data-path", default=str(DEFAULT_CONFIG.data_path))
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--output-dir", default=str(DEFAULT_CONFIG.artifacts_dir))
    return parser


if __name__ == "__main__":
    run_training(build_parser().parse_args())
