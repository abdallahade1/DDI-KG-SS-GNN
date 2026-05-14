# Drug-Drug Interaction Prediction with KG-SS-GNN

**[Live Streamlit Demo](https://huggingface.co/spaces/AbdulrahmanMahmoud007/DDI-KG-SS-GNN) | [Notebook on Kaggle](https://www.kaggle.com/code/abdallahadelabdallah/ddi-prediction)**
    

This repository contains the modular code version of `../ddi-phase2_v2.ipynb`.
It uses the local DrugBank DDI dataset in `../data/drugbank_ddi.csv` and trains
the final KG-SS-GNN model implemented in the updated notebook.

## Dataset

The expected local dataset file is:

```text
../data/drugbank_ddi.csv
```

It must contain the TDC DrugBank columns:

```text
Drug1_ID, Drug1, Drug2_ID, Drug2, Y
```

The downloaded file in this project has 191,808 interaction rows, 86 interaction
classes, and 1,706 unique drugs.

You can either place an already downloaded copy at `../data/drugbank_ddi.csv`
or import/download the dataset directly from Therapeutics Data Commons (TDC).
The repo code expects the CSV file after download, so run this once before
training if the file is missing:

```powershell
pip install PyTDC
python -c "from tdc.multi_pred import DDI; import os; os.makedirs('../data', exist_ok=True); data = DDI(name='DrugBank', path='../data'); df = data.get_data(); df.to_csv('../data/drugbank_ddi.csv', index=False); print(df.shape)"
```

Equivalent Python snippet:

```python
from tdc.multi_pred import DDI
import os

os.makedirs("../data", exist_ok=True)
data = DDI(name="DrugBank", path="../data")
df = data.get_data()
df.to_csv("../data/drugbank_ddi.csv", index=False)
print(df.shape)
```

## Model

The final model is `KGSSGNN`:

- 79-dimensional atom features from RDKit.
- 10-dimensional bond features from RDKit.
- BRICS/Morgan fingerprint features with 512 bits.
- 141-dimensional KG-proxy biological features.
- Shared 3-layer GAT encoder.
- Cross-drug co-attention.
- KG feature projection.
- MLP classifier over 86 DrugBank interaction types.

## Repository Layout

```text
repo/
  app.py
  requirements.txt
  README.md
  docs/
    DDI_Presentation.pdf
    drug_drug__interaction_prediction_technical_report.pdf
  src/ddi_project/
    checkpoints.py
    config.py
    data.py
    evaluate.py
    features.py
    models.py
    predict.py
    train.py
    utils.py
```

## Documentation

The `docs/` folder contains the project presentation and the final technical
report PDF. These files are included so the repository has both the runnable
code and the submitted project documentation in one place.

## Reproducibility Instructions

Create an environment and install dependencies. Use Python 3.12;
the RDKit, PyTorch, and PyTorch Geometric wheels may not install correctly on
newer interpreter versions such as Python 3.14.


Train the final model:

```powershell
python -m src.ddi_project.train --data-path ../data/drugbank_ddi.csv --epochs 100 --batch-size 64
```

The notebook checkpoints used by the Streamlit demo are expected at:

```text
../checkpoint/kgssgnn_best_pickle_cold_split.pkl
../checkpoint/kgssgnn_best_pickle_semi_cold_split.pkl
```

If you train from the repo, the generated checkpoint is saved to:

```text
artifacts/kgssgnn_best.pt
```

Evaluate the checkpoint:

```powershell
python -m src.ddi_project.evaluate --checkpoint ../checkpoint/kgssgnn_best_pickle_cold_split.pkl --data-path ../data/drugbank_ddi.csv
```

Run the Streamlit demo:

```powershell
streamlit run app.py
```

The demo lets you choose between the `cold_split` checkpoint,
`semi_cold_split` checkpoint, or a custom checkpoint path. It does not use a
synthetic fallback model.
