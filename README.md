# Airbus FARM Data Science Challenge

An end-to-end prototype for predicting airfoil self-noise and detecting anomalous
sound-pressure measurements caused by simulated instrument faults.

## Project status

The project is currently in the clean-data EDA phase. The notebook
[`notebooks/Airbus_FARM_Airfoil_EDA.ipynb`](notebooks/Airbus_FARM_Airfoil_EDA.ipynb)
provides:

- schema normalization and validation;
- data-type, cardinality, missing-value, and duplicate checks;
- descriptive statistics, skewness, and physical-value checks;
- Pearson and Spearman relationship analysis; and
- focused plots of the target, frequency response, angle-of-attack effect, and
  operating-regime interactions.

Model-development notebooks and a reusable Python package will follow in later
branches.

## Environment setup

Python 3.13 is recommended. Poetry is not required for the notebook phase.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab
```

Open `notebooks/Airbus_FARM_Airfoil_EDA.ipynb` and run all cells. The notebook
locates the source dataset whether Jupyter is started from the repository root
or the `notebooks/` directory.

## Data

The source workbook is expected at:

```text
data/airfoil_self_noise-dataset.xlsx
```

This EDA is read-only: it does not split, overwrite, or export transformed data.

> **Modelling note:** Synthetic anomalies should be injected only into a
> held-out test copy later. This EDA does not contaminate the clean source data.

## Dataset

Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise*.
UCI Machine Learning Repository. <https://doi.org/10.24432/C5VW2C>
