# Airbus FARM Data Science Challenge

An end-to-end prototype for predicting airfoil self-noise and detecting anomalous
sound-pressure measurements caused by simulated instrument faults.


## Environment setup

Python 3.13 is recommended. Poetry is not required for the notebook phase.


Open notebooks/eda.ipynb and run all cells. The notebook
locates the source dataset whether Jupyter is started from the repository root
or the notebooks directory.

## EDA and Data preparation

The source data is expected at folder: data/airfoil_self_noise-dataset.xlsx

The EDA is read-only: it does not split, overwrite, or export transformed data.
The data-preparation.ipynb notebook prepares the train, validation and test datasets, scales the features and target when necessary, and introduces different levels of noise to the test set.


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

## Model training

The models use these five physical features:

- frequency
- attack-angle
- chord-length
- free-stream-velocity
- suction-side-displacement-thickness

The target is sound pressure in dB.

Random Forest and XGBoost use the original feature values and do not require feature scaling. PyTorch and TabNet use scaled features. Their predictions are transformed back to dB before evaluation.

The notebooks contain the training, validation and test evaluations. The final inference script uses a saved model and does not repeat the training experiments.

## How to run inference

Install the dependencies with Poetry:

poetry install

Run inference from the repository root:

poetry run python inference/inference.py \
	--model models/random_forest.joblib \
	--data data/test_with_anomalies.csv

The results are saved to results/predictions.csv.

The input file must contain the five physical features and either:

- observed_sound_pressure_db
- scaled-sound-pressure

The output adds these columns:

- expected_sound_pressure_db
- residual_db
- anomaly_threshold_db
- predicted_anomaly

To test your own data, replace data/test_with_anomalies.csv with a CSV file that has the required columns, or pass its path with --data.

## Results and notebooks

Saved evaluation results are in the results folder. The notebooks show the data preparation, model training, validation and test evaluations.

The dataset is small, so it provides limited data for deep learning models. The deep learning results should therefore be interpreted with care.

## Use of AI for background support

AI was used for:

- debugging syntax errors
- researching and understanding WTT
- researching and adding simulated noise
- formatting plots
- testing the inference process


