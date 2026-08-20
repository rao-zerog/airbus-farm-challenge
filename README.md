# Airbus FARM Data Science Challenge

An end-to-end prototype for modelling expected airfoil sound pressure and detecting non-physical acoustic sensor anomalies caused by simulated instrument faults.

The project compares baseline machine-learning models with PyTorch-based deep-learning models. Random Forest was selected as the final model because it achieved the best validation and clean-test regression performance and the strongest anomaly-detection F1 score.

## Business problem

Wind-tunnel test campaigns can lose valuable test time when acoustic sensors produce transient faults or severe noise spikes. The prototype learns the expected clean sound-pressure level from the physical test conditions and compares that prediction with the observed sensor reading.

Every supplied observation is classified using the absolute prediction residual:

```text
residual = abs(observed sound pressure - expected sound pressure)
anomaly = residual > validation-derived threshold
```

The inference process does not retrain the model and does not call a generative-AI service at runtime.

## Dataset

The project uses the UCI Airfoil Self-Noise dataset, containing 1,503 clean wind-tunnel observations.

Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5VW2C](https://doi.org/10.24432/C5VW2C)

The source workbook is stored at:

```text
data/airfoil_self_noise-dataset.xlsx
```

The five input features are:

| Feature | Unit |
|---|---|
| `frequency` | Hz |
| `attack-angle` | degrees |
| `chord-length` | metres |
| `free-stream-velocity` | m/s |
| `suction-side-displacement-thickness` | metres |

The regression target is `scaled-sound-pressure`, measured in dB. Despite its source column name, this is the original sound-pressure target; the separately standardized target is called `target_standardized` in the prepared files.


## Environment setup

Python `>=3.12,<3.14` is required.

1. Create a virtual environment
python -m venv .venv

2. Activate the virtual environment

On macOS or Linux:
source .venv/bin/activate

On Windows PowerShell:
.venv\Scripts\Activate.ps1

3. Install Poetry and project dependencies
With the virtual environment active:
python -m pip install --upgrade pip
python -m pip install "poetry>=2,<3"
poetry install


## Reproduce the analysis

Run the notebooks in this order:

1. `notebooks/eda.ipynb`
2. `notebooks/data-preparation.ipynb`
3. `notebooks/model-training.ipynb`

The workflow performs the following steps:

1. Explores and validates the clean source data.
2. Creates a reproducible 70% training, 15% validation, and 15% test split using `random_state=42`.
3. Fits feature and target scalers using only the training split.
4. Injects synthetic faults into 15% of a separate test-data copy.
5. Trains and compares the regression models.
6. Selects the final model using clean validation RMSE.
7. Calculates a separate anomaly threshold for each model from clean validation residuals.
8. Evaluates regression accuracy and synthetic-anomaly detection on held-out test data.

The model-training notebook creates the selected artifact:

```text
models/random_forest.joblib
```

## Models evaluated

The project compares:

- Linear Regression baseline
- Random Forest
- XGBoost
- Vanilla PyTorch MLP baseline
- TabNetRegressor

TabPFNRegressor was considered as future work but was not implemented or evaluated.

Random Forest and XGBoost use the original feature values. Linear Regression uses standardized features while retaining the target in dB. The PyTorch MLP and TabNet use standardized features and a standardized target; their predictions are transformed back to dB before evaluation.

Random Forest and XGBoost were tuned with three-fold `GridSearchCV`. The PyTorch models used fixed architectures and early stopping.

## Model results

### Clean-test regression

| Model | MAE (dB) | RMSE (dB) | R² |
|---|---:|---:|---:|
| **Random Forest** | **1.352** | **1.845** | **0.924** |
| TabNetRegressor | 1.450 | 2.010 | 0.909 |
| XGBoost | 1.670 | 2.235 | 0.888 |
| Vanilla PyTorch MLP | 2.068 | 2.904 | 0.811 |
| Linear Regression | 3.642 | 4.675 | 0.509 |

### Synthetic-anomaly detection

Each model uses the 99th percentile of its absolute clean-validation residuals as its anomaly threshold. The test labels are not used to choose the threshold.

| Model | Threshold (dB) | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Random Forest** | **6.137462** | **1.000** | **0.706** | **0.828** |
| TabNetRegressor | 6.240543 | 0.920 | 0.676 | 0.780 |
| XGBoost | 7.252749 | 0.957 | 0.647 | 0.772 |
| Vanilla PyTorch MLP | 9.975262 | 0.947 | 0.529 | 0.679 |
| Linear Regression | 14.027633 | 0.909 | 0.294 | 0.444 |

Random Forest detected 24 of 34 injected anomalies with no false positives. Its recall by severity was:

| Severity | Detected | Total | Recall |
|---|---:|---:|---:|
| Low | 3 | 11 | 0.273 |
| Moderate | 9 | 11 | 0.818 |
| Severe | 12 | 12 | 1.000 |

## Random Forest inference

### Required input

The input CSV or Parquet file must contain the five original, unscaled feature columns:

- `frequency`
- `attack-angle`
- `chord-length`
- `free-stream-velocity`
- `suction-side-displacement-thickness`

It must also contain an observed sound-pressure column. The default name is `observed_sound_pressure_db`. If the column uses another name, pass `--observed-column`.

The original `scaled-sound-pressure` name is automatically accepted when the default observed column is absent.

Do not scale Random Forest inputs

The Random Forest model was trained using original values in their physical units, so inference must use the same representation.

```text
Random Forest training:  original features
Random Forest inference: original features
Feature scaler required: no
```

Do not pass columns ending in `_standardized`, and do not apply `feature_scaler.joblib` before Random Forest inference.

### Run with the saved model

From the repository root:

```bash
poetry run python inference/inference.py \
  --model models/random_forest.joblib \
  --data data/test_with_anomalies.csv
```

The default output is:

```text
results/predictions.csv
```


### Output columns

The original input columns are preserved and these columns are added:

- `expected_sound_pressure_db`
- `residual_db`
- `anomaly_threshold_db`
- `predicted_anomaly` (`0` for normal, `1` for anomalous)

## Decision log

- **Data split:** A fixed 70/15/15 split keeps training, model selection, and final testing separate.
- **Scaling:** Scalers are fitted only on training data. Tree models use original features; neural networks use standardized features and targets.
- **Synthetic faults:** Faults are injected only into a test-data copy, leaving training and validation data clean.
- **Fault severity:** Positive spikes of 3–5 dB, 6–10 dB, and 12–20 dB represent low, moderate, and severe faults.
- **Model selection:** The final model is selected using clean validation RMSE, not test performance.
- **Threshold selection:** The anomaly threshold is the 99th percentile of clean validation residuals and is frozen before anomaly-test evaluation.
- **Final model:** Random Forest was selected because it achieved the best validation RMSE and the strongest anomaly F1 score on this dataset.

## Limitations and future work

- The dataset is small and represents already-clean historical measurements.
- Synthetic faults are positive additive spikes; negative spikes, drift, dropouts, and correlated sensor faults are not yet represented.
- Low-severity anomaly recall is limited and could be improved by validation-only threshold tuning, with a likely increase in false positives.
- This prototype processes files in batch. A real-time deployment would require streaming integration, monitoring, and latency testing.
- The anomaly threshold should be monitored and recalibrated when the operating distribution changes.
- TabPFN and additional PyTorch architectures could be evaluated in future work.

## Use of generative AI

AI-assisted tools were used for:

- Background research on wind-tunnel testing and synthetic measurement faults
- Debugging syntax and environment issues
- Reviewing plotting and evaluation code
- Testing the inference workflow
- Improving documentation structure and wording

The author manually reviewed and refactored the final workflow, including the train/validation/test separation, train-only scaler fitting, model-specific preprocessing, validation-derived anomaly thresholds, model selection, saved artifacts, inference inputs, and reported conclusions.
