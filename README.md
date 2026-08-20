# Airbus FARM Data Science Challenge

An end-to-end prototype for predicting airfoil self-noise and detecting anomalous
sound-pressure measurements caused by simulated instrument faults.

## Project status

- EDA
- Data preparation


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

## Models 

This work compares the performance of deep learning and machine learning approaches.

- Baseline ML - Linear Regression
- Candidate model - XGBoost, Randomforest
- Baseline DL - vanilla Pytorch models 
- Candidate models - TabNetRegressor, TabPFNRegressor

Multiple candidate models were trained and evaluated to compare predictive performance. The best-performing model was selected using the validation set and packaged as the single inference model. This provides a simple and reproducible interface for users while retaining the experimental comparison in the training notebook.
