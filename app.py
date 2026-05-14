from pathlib import Path

import pandas as pd
import streamlit as st

from src.ddi_project.config import DEFAULT_CONFIG


st.set_page_config(page_title="KG-SS-GNN DDI Demo", page_icon="DDI", layout="wide")

st.title("KG-SS-GNN Drug-Drug Interaction Prediction")
st.caption("Inference demo using the final KG-SS-GNN architecture and local DrugBank SMILES inputs.")


EXAMPLES = {
    "Aspirin + Warfarin": (
        "Aspirin",
        "CC(=O)Oc1ccccc1C(=O)O",
        "Warfarin",
        "CC1=C(C(=O)Cc2ccccc21)OCC(=O)O",
    ),
    "Ibuprofen + Methotrexate": (
        "Ibuprofen",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Methotrexate",
        "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O",
    ),
    "Custom": ("Drug A", "", "Drug B", ""),
}

CHECKPOINT_OPTIONS = {
    "cold_split": "../checkpoint/kgssgnn_best_pickle_cold_split.pkl",
    "semi_cold_split": "../checkpoint/kgssgnn_best_pickle_semi_cold_split.pkl",
    "Custom path": str(DEFAULT_CONFIG.checkpoint_path),
}


@st.cache_resource
def load_predictor(checkpoint_path: str):
    from src.ddi_project.predict import KGSSGNNPredictor

    return KGSSGNNPredictor(checkpoint=checkpoint_path)


def molecule_image(smiles: str, size: tuple[int, int] = (420, 260)):
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Draw.MolToImage(mol, size=size)


def brics_fragment_images(smiles: str, max_fragments: int = 8):
    from rdkit import Chem
    from rdkit.Chem import BRICS, Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fragments = sorted(BRICS.BRICSDecompose(mol))
    fragment_mols = []
    legends = []
    for fragment in fragments:
        fragment_mol = Chem.MolFromSmiles(fragment)
        if fragment_mol is not None:
            fragment_mols.append(fragment_mol)
            legends.append(fragment[:24])
        if len(fragment_mols) == max_fragments:
            break

    if not fragment_mols:
        return None, 0
    image = Draw.MolsToGridImage(
        fragment_mols,
        molsPerRow=4,
        subImgSize=(220, 150),
        legends=legends,
    )
    return image, len(fragments)


def descriptor_table(desc_a: dict, desc_b: dict, name_a: str, name_b: str) -> pd.DataFrame:
    labels = {
        "mol_weight": "Molecular weight",
        "logp": "LogP",
        "hbd": "H-bond donors",
        "hba": "H-bond acceptors",
        "tpsa": "TPSA",
        "rotatable_bonds": "Rotatable bonds",
        "aromatic_rings": "Aromatic rings",
        "atoms": "Atoms",
        "bonds": "Bonds",
    }
    rows = []
    for key, label in labels.items():
        rows.append(
            {
                "descriptor": label,
                name_a: round(float(desc_a[key]), 3),
                name_b: round(float(desc_b[key]), 3),
            }
        )
    return pd.DataFrame(rows)


def render_pipeline_diagram() -> None:
    stages = [
        ("1", "SMILES", "User input"),
        ("2", "RDKit", "parse molecules"),
        ("3", "Features", "graph + fingerprint + KG"),
        ("4", "KG-SS-GNN", "GAT + co-attention"),
        ("5", "Classifier", "86 interaction classes"),
    ]
    cols = st.columns(len(stages))
    for col, (number, title, caption) in zip(cols, stages):
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #ddd;border-radius:8px;padding:12px;text-align:center;min-height:112px">
                    <div style="font-size:0.8rem;color:#8a4b3a">Step {number}</div>
                    <div style="font-weight:700;font-size:1.05rem;margin-top:4px">{title}</div>
                    <div style="font-size:0.85rem;color:#666;margin-top:6px">{caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


checkpoint_choice = st.sidebar.selectbox("Checkpoint", list(CHECKPOINT_OPTIONS.keys()))
default_checkpoint_path = CHECKPOINT_OPTIONS[checkpoint_choice]
if checkpoint_choice == "Custom path":
    checkpoint_path = st.sidebar.text_input("Checkpoint path", value=default_checkpoint_path)
else:
    checkpoint_path = default_checkpoint_path
    st.sidebar.text_input("Checkpoint path", value=checkpoint_path, disabled=True)

checkpoint = Path(checkpoint_path)
if not checkpoint.exists():
    st.error(
        "KG-SS-GNN checkpoint is missing. Train the model first or copy the notebook output "
        "into `project/checkpoint`."
    )
    st.code("python -m src.ddi_project.train --data-path ../data/drugbank_ddi.csv --epochs 100")
    st.stop()

try:
    predictor = load_predictor(checkpoint_path)
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    st.error(f"Missing Python dependency: `{missing}`.")
    st.code("pip install -r requirements.txt")
    st.stop()
except ImportError as exc:
    st.error(f"Dependency import failed: {exc}")
    st.code("pip install -r requirements.txt")
    st.stop()

selected = st.selectbox("Example pair", list(EXAMPLES.keys()))
default_a_name, default_a_smiles, default_b_name, default_b_smiles = EXAMPLES[selected]

left, right = st.columns(2)
with left:
    drug_a_name = st.text_input("Drug A name", value=default_a_name, key=f"{selected}_drug_a_name")
    smiles_a = st.text_area("Drug A SMILES", value=default_a_smiles, height=110, key=f"{selected}_smiles_a")
with right:
    drug_b_name = st.text_input("Drug B name", value=default_b_name, key=f"{selected}_drug_b_name")
    smiles_b = st.text_area("Drug B SMILES", value=default_b_smiles, height=110, key=f"{selected}_smiles_b")

if st.button("Predict interaction", type="primary"):
    try:
        from rdkit import Chem
        from src.ddi_project.config import DEFAULT_CONFIG
        from src.ddi_project.features import fingerprint, kg_features, smiles_to_graph

        clean_smiles_a = smiles_a.strip()
        clean_smiles_b = smiles_b.strip()
        result = predictor.predict(smiles_a.strip(), smiles_b.strip(), top_k=5)
        st.subheader(f"Predictions for {drug_a_name} + {drug_b_name}")
        pred_df = pd.DataFrame(result["predictions"])
        pred_df["probability"] = pred_df["probability"].map(lambda value: round(value, 4))
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        st.bar_chart(pred_df.set_index("label"))

        with st.expander("Show inference visualizations", expanded=True):
            st.markdown("#### Model pipeline")
            render_pipeline_diagram()

            st.markdown("#### 1. SMILES parsing and molecule rendering")
            mol_a = Chem.MolFromSmiles(clean_smiles_a)
            mol_b = Chem.MolFromSmiles(clean_smiles_b)
            img_left, img_right = st.columns(2)
            with img_left:
                st.image(molecule_image(clean_smiles_a), caption=f"{drug_a_name}")
            with img_right:
                st.image(molecule_image(clean_smiles_b), caption=f"{drug_b_name}")

            st.markdown("#### 2. Molecular graph construction")
            graph_a = smiles_to_graph(clean_smiles_a)
            graph_b = smiles_to_graph(clean_smiles_b)
            gcols = st.columns(4)
            gcols[0].metric(f"{drug_a_name} atoms", int(graph_a.num_nodes))
            gcols[1].metric(f"{drug_a_name} directed edges", int(graph_a.edge_index.shape[1]))
            gcols[2].metric(f"{drug_b_name} atoms", int(graph_b.num_nodes))
            gcols[3].metric(f"{drug_b_name} directed edges", int(graph_b.edge_index.shape[1]))
            st.caption(
                f"Atom feature vectors have {DEFAULT_CONFIG.atom_feature_dim} dimensions; "
                f"bond feature vectors have {DEFAULT_CONFIG.bond_feature_dim} dimensions."
            )

            st.markdown("#### 3. BRICS fragments and Morgan fingerprint density")
            fp_a = fingerprint(clean_smiles_a)
            fp_b = fingerprint(clean_smiles_b)
            fp_df = pd.DataFrame(
                [
                    {
                        "drug": drug_a_name,
                        "active_bits": int(fp_a.sum().item()),
                        "inactive_bits": int(DEFAULT_CONFIG.fp_dim - fp_a.sum().item()),
                    },
                    {
                        "drug": drug_b_name,
                        "active_bits": int(fp_b.sum().item()),
                        "inactive_bits": int(DEFAULT_CONFIG.fp_dim - fp_b.sum().item()),
                    },
                ]
            )
            st.bar_chart(fp_df.set_index("drug")[["active_bits", "inactive_bits"]])
            st.caption(f"Each BRICS/Morgan fingerprint has {DEFAULT_CONFIG.fp_dim} bits.")

            frag_left, frag_right = st.columns(2)
            with frag_left:
                fragment_image, fragment_count = brics_fragment_images(clean_smiles_a)
                st.markdown(f"**{drug_a_name} BRICS fragments: {fragment_count}**")
                if fragment_image is not None:
                    st.image(fragment_image)
                else:
                    st.info("No displayable BRICS fragments.")
            with frag_right:
                fragment_image, fragment_count = brics_fragment_images(clean_smiles_b)
                st.markdown(f"**{drug_b_name} BRICS fragments: {fragment_count}**")
                if fragment_image is not None:
                    st.image(fragment_image)
                else:
                    st.info("No displayable BRICS fragments.")

            st.markdown("#### 4. KG-proxy descriptor comparison")
            desc_df = descriptor_table(
                result["descriptors_a"],
                result["descriptors_b"],
                drug_a_name,
                drug_b_name,
            )
            st.dataframe(desc_df, use_container_width=True, hide_index=True)
            chart_df = desc_df[
                desc_df["descriptor"].isin(
                    ["LogP", "H-bond donors", "H-bond acceptors", "Rotatable bonds", "Aromatic rings"]
                )
            ].set_index("descriptor")
            st.bar_chart(chart_df)
            kg_a = kg_features(clean_smiles_a)
            kg_b = kg_features(clean_smiles_b)
            kcols = st.columns(2)
            kcols[0].metric(f"{drug_a_name} KG-proxy dimensions", int(kg_a.numel()))
            kcols[1].metric(f"{drug_b_name} KG-proxy dimensions", int(kg_b.numel()))

            st.markdown("#### 5. Prediction confidence")
            top_probability = float(pred_df.iloc[0]["probability"])
            second_probability = float(pred_df.iloc[1]["probability"]) if len(pred_df) > 1 else 0.0
            ccols = st.columns(3)
            ccols[0].metric("Top class", pred_df.iloc[0]["label"])
            ccols[1].metric("Top probability", f"{top_probability:.4f}")
            ccols[2].metric("Top-1 / Top-2 gap", f"{top_probability - second_probability:.4f}")
    except Exception as exc:
        st.error(str(exc))
