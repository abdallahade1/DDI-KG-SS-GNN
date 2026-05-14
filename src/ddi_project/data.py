from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Batch, Data
from torch.utils.data import WeightedRandomSampler

from .config import DEFAULT_CONFIG
from .features import fingerprint, kg_features, smiles_to_graph


def empty_molecular_graph() -> Data:
    return Data(
        x=torch.zeros((1, DEFAULT_CONFIG.atom_feature_dim), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, DEFAULT_CONFIG.bond_feature_dim), dtype=torch.float32),
        num_nodes=1,
    )


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    smiles_map: dict[str, str]
    label_encoder: LabelEncoder
    n_classes: int


def load_local_drugbank(
    csv_path: str | Path = DEFAULT_CONFIG.data_path,
    seed: int = DEFAULT_CONFIG.seed,
    test_drug_fraction: float = 0.20,
    valid_pair_fraction: float = 0.10,
) -> DatasetBundle:
    """Load local TDC DrugBank CSV and create a cold-drug split.

    The expected CSV columns are Drug1_ID, Drug1, Drug2_ID, Drug2, and Y.
    Test rows contain at least one drug from the held-out cold-drug set.
    Validation rows are sampled from the remaining non-cold rows.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DrugBank CSV not found at {path.resolve()}. "
            "Expected project/data/drugbank_ddi.csv."
        )

    df = pd.read_csv(path)
    required = {"Drug1_ID", "Drug1", "Drug2_ID", "Drug2", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.rename(columns={"Y": "label"}).dropna(subset=["Drug1", "Drug2", "label"])
    df["label"] = df["label"].astype(int)

    smiles_map: dict[str, str] = {}
    for _, row in df.iterrows():
        smiles_map[str(row["Drug1_ID"])] = row["Drug1"]
        smiles_map[str(row["Drug2_ID"])] = row["Drug2"]

    rng = np.random.default_rng(seed)
    all_drugs = np.array(sorted(smiles_map))
    n_cold = max(1, int(len(all_drugs) * test_drug_fraction))
    cold_drugs = set(rng.choice(all_drugs, size=n_cold, replace=False).tolist())

    cold_mask = df["Drug1_ID"].isin(cold_drugs) | df["Drug2_ID"].isin(cold_drugs)
    test = df[cold_mask].copy()
    remaining = df[~cold_mask].copy()

    valid = (
        remaining.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(frac=valid_pair_fraction, random_state=seed) if len(group) > 1 else group)
        .copy()
    )
    train = remaining.drop(valid.index).copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"])
    for frame in (train, valid, test):
        frame["label"] = label_encoder.transform(frame["label"])

    return DatasetBundle(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
        smiles_map=smiles_map,
        label_encoder=label_encoder,
        n_classes=len(label_encoder.classes_),
    )


def precompute_features(smiles_map: dict[str, str]) -> tuple[dict[str, Data], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    graphs: dict[str, Data] = {}
    fps: dict[str, torch.Tensor] = {}
    kgs: dict[str, torch.Tensor] = {}
    for drug_id, smiles in smiles_map.items():
        try:
            graphs[drug_id] = smiles_to_graph(smiles)
            fps[drug_id] = fingerprint(smiles, DEFAULT_CONFIG.fp_dim)
            kgs[drug_id] = kg_features(smiles)
        except ValueError:
            graphs[drug_id] = empty_molecular_graph()
            fps[drug_id] = torch.zeros(DEFAULT_CONFIG.fp_dim, dtype=torch.float32)
            kgs[drug_id] = torch.zeros(DEFAULT_CONFIG.kg_dim, dtype=torch.float32)
    return graphs, fps, kgs


class KGSSGNNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        graph_cache: dict[str, Data],
        fp_cache: dict[str, torch.Tensor],
        kg_cache: dict[str, torch.Tensor],
    ):
        self.frame = frame.reset_index(drop=True)
        self.graph_cache = graph_cache
        self.fp_cache = fp_cache
        self.kg_cache = kg_cache

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        drug_a = str(row["Drug1_ID"])
        drug_b = str(row["Drug2_ID"])
        return (
            self.graph_cache[drug_a],
            self.graph_cache[drug_b],
            self.fp_cache[drug_a],
            self.fp_cache[drug_b],
            self.kg_cache[drug_a],
            self.kg_cache[drug_b],
            int(row["label"]),
        )


def collate_kgssgnn(batch):
    graph_a, graph_b, fp_a, fp_b, kg_a, kg_b, y = zip(*batch)
    return (
        Batch.from_data_list(list(graph_a)),
        Batch.from_data_list(list(graph_b)),
        torch.stack(fp_a),
        torch.stack(fp_b),
        torch.stack(kg_a),
        torch.stack(kg_b),
        torch.tensor(y, dtype=torch.long),
    )


def weighted_sampler(labels: np.ndarray, n_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    counts = np.where(counts == 0, 1, counts)
    sample_weights = (1.0 / counts)[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
