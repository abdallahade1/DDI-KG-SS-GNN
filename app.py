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
    "New checkpoint": "../checkpoint/kgssgnn_best_pickle_new.pkl",
    "Old checkpoint": "../checkpoint/kgssgnn_best_pickle_old.pkl",
    "Custom path": str(DEFAULT_CONFIG.checkpoint_path),
}


@st.cache_resource
def load_predictor(checkpoint_path: str):
    from src.ddi_project.predict import KGSSGNNPredictor

    return KGSSGNNPredictor(checkpoint=checkpoint_path)


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
    drug_a_name = st.text_input("Drug A name", value=default_a_name)
    smiles_a = st.text_area("Drug A SMILES", value=default_a_smiles, height=110)
with right:
    drug_b_name = st.text_input("Drug B name", value=default_b_name)
    smiles_b = st.text_area("Drug B SMILES", value=default_b_smiles, height=110)

if st.button("Predict interaction", type="primary"):
    try:
        result = predictor.predict(smiles_a.strip(), smiles_b.strip(), top_k=5)
        st.subheader(f"Predictions for {drug_a_name} + {drug_b_name}")
        pred_df = pd.DataFrame(result["predictions"])
        pred_df["probability"] = pred_df["probability"].map(lambda value: round(value, 4))
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        st.bar_chart(pred_df.set_index("label"))

        dleft, dright = st.columns(2)
        with dleft:
            st.markdown("**Drug A descriptors**")
            st.json(result["descriptors_a"])
        with dright:
            st.markdown("**Drug B descriptors**")
            st.json(result["descriptors_b"])
    except Exception as exc:
        st.error(str(exc))
