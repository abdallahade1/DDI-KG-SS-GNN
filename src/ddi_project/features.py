from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch_geometric.data import Data

from .config import DEFAULT_CONFIG


def validate_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def molecular_descriptors(smiles: str) -> dict[str, float]:
    mol = validate_smiles(smiles)
    return {
        "mol_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "atoms": float(mol.GetNumAtoms()),
        "bonds": float(mol.GetNumBonds()),
    }


def atom_features(atom) -> list[float]:
    def one_hot(value, choices):
        values = [0.0] * len(choices)
        values[choices.index(value) if value in choices else -1] = 1.0
        return values

    atomic_nums = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53, 0]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "other",
    ]
    chiralities = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        "other",
    ]

    features = one_hot(atom.GetAtomicNum(), atomic_nums)
    features += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    features += one_hot(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6])
    features += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    features += one_hot(atom.GetHybridization(), hybridizations)
    features += one_hot(atom.GetChiralTag(), chiralities)
    features += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features += [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.IsInRingSize(3)),
        float(atom.IsInRingSize(4)),
        float(atom.IsInRingSize(5)),
        float(atom.IsInRingSize(6)),
        atom.GetMass() / 100.0,
        float(atom.GetNoImplicit()),
    ]
    features += [0.0] * (DEFAULT_CONFIG.atom_feature_dim - len(features))
    return features[: DEFAULT_CONFIG.atom_feature_dim]


def bond_features(bond) -> list[float]:
    def one_hot(value, choices):
        values = [0.0] * len(choices)
        values[choices.index(value) if value in choices else -1] = 1.0
        return values

    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        "other",
    ]
    features = one_hot(bond.GetBondType(), bond_types)
    features += [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
    ]
    features += [0.0] * (DEFAULT_CONFIG.bond_feature_dim - len(features))
    return features[: DEFAULT_CONFIG.bond_feature_dim]


def smiles_to_graph(smiles: str) -> Data:
    mol = validate_smiles(smiles)
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        attrs = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([attrs, attrs])

    if not edge_index:
        return Data(
            x=x,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, DEFAULT_CONFIG.bond_feature_dim), dtype=torch.float32),
            num_nodes=x.size(0),
        )

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        num_nodes=x.size(0),
    )


def fingerprint(smiles: str, fp_dim: int = DEFAULT_CONFIG.fp_dim) -> torch.Tensor:
    mol = validate_smiles(smiles)
    if mol.GetNumAtoms() > 100:
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
        return torch.tensor(np.asarray(bitvect, dtype=np.float32), dtype=torch.float32)

    pooled = np.zeros(fp_dim, dtype=np.float32)
    fragments = list(BRICS.BRICSDecompose(mol)) or [smiles]
    for fragment in fragments:
        fragment_mol = Chem.MolFromSmiles(fragment)
        if fragment_mol is None:
            continue
        bitvect = GetMorganFingerprintAsBitVect(fragment_mol, 2, fp_dim)
        pooled = np.maximum(pooled, np.asarray(bitvect, dtype=np.float32))
    return torch.tensor(pooled, dtype=torch.float32)


def kg_features(smiles: str) -> torch.Tensor:
    mol = validate_smiles(smiles)
    fp_r1 = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=64), dtype=np.float32)
    fp_r3 = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=64), dtype=np.float32)
    physchem = np.array(
        [
            Descriptors.MolWt(mol) / 600.0,
            Descriptors.MolLogP(mol) / 10.0,
            Descriptors.TPSA(mol) / 150.0,
            Descriptors.NumHDonors(mol) / 10.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 15.0,
            rdMolDescriptors.CalcNumAromaticRings(mol) / 6.0,
            rdMolDescriptors.CalcNumRings(mol) / 8.0,
            rdMolDescriptors.CalcNumHeterocycles(mol) / 6.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.NumRadicalElectrons(mol) / 4.0,
            min(mol.GetNumAtoms() / 80.0, 1.0),
            min(mol.GetNumBonds() / 90.0, 1.0),
        ],
        dtype=np.float32,
    )
    return torch.tensor(np.concatenate([fp_r1, fp_r3, physchem]), dtype=torch.float32)
