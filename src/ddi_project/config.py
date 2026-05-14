from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    atom_feature_dim: int = 79
    bond_feature_dim: int = 10
    fp_dim: int = 512
    kg_dim: int = 141
    hidden_dim: int = 128
    gat_heads: int = 4
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    artifacts_dir: Path = Path("artifacts")
    data_path: Path = Path("../data/drugbank_ddi.csv")
    checkpoint_path: Path = Path("../checkpoint/kgssgnn_best_pickle_new.pkl")


DEFAULT_CONFIG = ProjectConfig()
