# Airbus FARM Data Science Challenge

An end-to-end prototype for predicting airfoil self-noise and detecting anomalous
sound-pressure measurements caused by simulated instrument faults.

## Project status

The project is currently in the clean-data EDA phase. The notebook - eda.ipynb
provides:

- schema normalization and validation;
- data-type, cardinality, missing-value, and duplicate checks;
- descriptive statistics, skewness, and physical-value checks;
- Pearson and Spearman relationship analysis; and
- focused plots of the target, frequency response, angle-of-attack effect, and
  operating-regime interactions.


## Environment setup

Python 3.13 is recommended. Poetry is not required for the notebook phase.


Open `notebooks/Airbus_FARM_Airfoil_EDA.ipynb` and run all cells. The notebook
locates the source dataset whether Jupyter is started from the repository root
or the `notebooks/` directory.

## EDA and Data preparation

The source data is expected at folder: data/airfoil_self_noise-dataset.xlsx

The EDA is read-only: it does not split, overwrite, or export transformed data.
The data-preparation.ipynb notebook prepares the train, val and test datasets, scales the features and target for use when necessary. Finally it intriduces noise of different ranges to the test set


## Dataset

Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise*.
UCI Machine Learning Repository. <https://doi.org/10.24432/C5VW2C>
