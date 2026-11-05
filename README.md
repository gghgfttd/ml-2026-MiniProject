# Bike-Sharing Demand Prediction — Machine Learning Mini-Project

**Course:** Machine Learning, National University of Kyiv-Mohyla Academy  
**Project type:** Empirical regression study  
**Target:** Hourly bike rental count (`cnt`)

## 1. Project Overview

This project studies how the choice of validation strategy affects the estimated predictive performance of machine-learning models on hourly bike-sharing data.

The central research question is:

> **How does estimated model performance change under random, blocked, and chronological validation, and does the size of this difference depend on the model family?**

The motivation is that hourly observations are not independent in time. Demand has daily and weekly patterns, and its overall level changes between 2011 and 2012. A random split can therefore evaluate a different task from predicting genuinely later observations: its training data may contain observations later than those in its test fold.

We compare three validation strategies and examine the resulting performance estimates, model errors, and systematic prediction patterns. We also test whether a logarithmic target transformation or a Poisson loss changes the error profile.

**This is a supervised regression study with time-aware evaluation, not a lag-based time-series forecasting model.** The main experiments use calendar and weather features; they do not use previous rental counts (`cnt`) as input features.

## 2. Dataset and Preprocessing

The project uses the hourly file (`data/hour.csv`) from the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

- **Observations:** 17,379 recorded hours.
- **Time coverage:** 2011–2012.
- **Location:** Washington, D.C.
- **Target:** `cnt`, the total number of rentals in an hour.
- **Task:** regression of hourly rental counts.

The preprocessing decisions are documented in `01_eda.ipynb` and implemented in `src/data.py`:

- `casual` and `registered` are removed because their sum gives the target `cnt` exactly; retaining them would cause target leakage.
- `instant` and `dteday` are not used as predictors. A timestamp is constructed and used to sort and index the observations chronologically.
- The year indicator `yr` is excluded from the main experiments, so models do not receive the 2011/2012 label directly.
- `atemp` and `season` are excluded from the main feature set; `temp` is retained.
- The 22 zero values in `hum` are treated as missing values.
- The three observations with `weathersit == 4` are grouped with category 3.
- Missing timestamps are not filled with invented rental counts.

The shared loader provides nine predictors: `hr`, `mnth`, `weekday`, `holiday`, `workingday`, `weathersit`, `temp`, `hum`, and `windspeed`.

## 3. Experimental Design

### Baselines

Two reference models establish a simple performance benchmark:

1. **Constant baseline:** predicts the mean rental count in the training fold.
2. **Hour × day-type baseline:** predicts the training-fold mean for the corresponding hour and day type, falling back to the training mean for an unseen combination.

Baseline predictions are calculated using the training portion of each fold; test-fold target values are not used to construct them.

### Candidate models

The project compares three model families, with two alternative variants in two of those families:

| Model family | Variants | What is examined |
|---|---|---|
| Ridge Regression | `ridge`, `ridge_log` | Regularised linear regression, with and without a `log1p` transformation of the target. |
| Random Forest | `rf` | A tree-ensemble model that can capture nonlinear relationships and feature interactions. |
| Histogram-based Gradient Boosting | `hgb`, `hgb_poisson` | Boosted trees trained with squared-error and Poisson losses. |

The variants are not counted as separate model families. The comparison therefore covers three families and five candidate-model variants, in addition to the two baselines.

### Model-specific preprocessing

Preprocessing is fitted within the relevant training fold:

- **Ridge:** categorical encoding, numerical imputation and scaling, a quadratic temperature feature, and an hour-by-working-day interaction. The `ridge_log` variant applies `log1p` to the target during fitting and transforms predictions back with `expm1`.
- **Random Forest:** tree-based model without feature scaling or one-hot encoding; missing humidity values are imputed within the pipeline.
- **HistGradientBoosting:** uses categorical features and native handling of missing values. Automatic early stopping is disabled to avoid introducing an additional random internal validation split into the time-ordered experiments.

The exact transformations and hyperparameter grids are documented in the corresponding notebooks.

## 4. Validation Strategy

The shared implementation in `src/validation.py` provides three schemes:

1. **Random K-Fold:** observations are shuffled before being divided into folds. A training fold can contain observations later in time than observations in its test fold.
2. **Blocked K-Fold:** observations remain in chronological order and each test fold is a contiguous block. However, the training fold contains observations both before and after that block. It is therefore not a simulation of forecasting only into the future.
3. **Chronological expanding-window validation:** the first 50% of the time span is used as the initial training period. The remaining time span is divided into five consecutive test periods of equal calendar duration. Each fold is trained only on observations earlier than its test period.

### Nested cross-validation

For experiments with hyperparameter tuning, the evaluation uses two nested loops:

- The **outer loop** estimates performance. It uses five splits and keeps each outer test fold out of the corresponding model-selection process.
- The **inner loop** selects hyperparameters using only the outer training data. It uses three splits with the same validation scheme as the outer loop.

For each outer fold, the search selects hyperparameters from the inner validation results and refits the selected estimator on the full outer training set. The refitted estimator is then evaluated on the untouched outer test fold. This process is repeated for all five outer folds.

The chronological scheme is the primary evaluation setting because the intended use is prediction for later time periods. Random and blocked validation are included to investigate how alternative split designs change the reported performance.

**Important distinction:** the inner validation folds select hyperparameters; the outer test folds estimate the performance of the model-selection procedure. The chronological outer test predictions cover 8,722 recorded hours in 2012. The initial chronological training period is not part of those outer test predictions.

## 5. Evaluation Metrics

The experiments report four regression metrics:

- **RMSE (Root Mean Squared Error):** measured in rentals per hour; penalises large errors more strongly. It is the primary hyperparameter-search metric.
- **MAE (Mean Absolute Error):** average absolute prediction error, measured in rentals per hour.
- **R²:** compares the model's squared prediction error with the error of predicting the mean target value in the evaluated sample.
- **RMSLE (Root Mean Squared Logarithmic Error):** measures errors on a logarithmic scale and gives a complementary view of relative errors across demand levels.

Negative predictions are retained when calculating RMSE, MAE, and R². For RMSLE only, negative predictions are clipped to zero before applying the logarithm. The frequency of negative predictions is analysed separately.

## 6. Main Results

### Performance on the same 2012 timestamps

To make the validation schemes easier to compare, the global analysis also evaluates predictions on the **same 8,722 chronological test timestamps in 2012**. The target observations are aligned across schemes and models. This makes the evaluated hours comparable, but **does not make the training sets identical**: random and blocked validation may use observations later than a given test timestamp, while chronological validation may use only earlier observations.

The table reports RMSE in rentals per hour:

| Model | Random | Blocked | Chronological |
|---|---:|---:|---:|
| Constant baseline | 213.8 | 219.0 | 220.9 |
| Hour × day-type baseline | 120.0 | 134.0 | 139.4 |
| Ridge | 96.3 | 126.6 | 128.7 |
| Ridge (`log1p` target) | 94.3 | 136.2 | 129.1 |
| Random Forest | 71.3 | 120.5 | **117.6** |
| HistGradientBoosting (squared error) | 68.3 | 119.0 | 121.3 |
| HistGradientBoosting (Poisson) | **67.2** | 121.2 | 121.9 |

The main findings are:

- Random validation gives substantially lower RMSE than chronological validation for the learned models. For example, the Random Forest RMSE rises from **71.3** under random validation to **117.6** under chronological validation; for the two boosting variants, it rises from **68.3 to 121.3** and from **67.2 to 121.9**, respectively.
- The size of this difference varies by model. It is comparatively small for the constant baseline (+3%) and larger for the learned models: +34% for Ridge, +65% for Random Forest, +78% for squared-error boosting, and +81% for Poisson boosting. Percentages are calculated relative to the random-validation RMSE on the matched timestamps.
- Most of the increase appears between random and blocked validation. Blocked and chronological RMSE are much closer in this matched comparison, although the schemes still represent different training information and prediction settings.
- **Random Forest has the lowest chronological RMSE (117.6)** among the evaluated candidate models. It improves on the hour × day-type baseline (139.4) by approximately **15.6%** in RMSE on the matched 2012 evaluation.
- Poisson loss gives a lower chronological RMSLE than squared-error boosting, but it does not improve chronological RMSE in this experiment (121.9 versus 121.3).

These results support the central conclusion: validation design materially affects the performance estimate, and the size of the effect is model-dependent. Random validation is not an appropriate estimate of performance when the intended task is prediction on later, unseen time periods.

### Error analysis

The comparison notebook also examines residuals and prediction errors by time, demand level, and weather. A recurring pattern is underprediction during high-demand working-day peaks. For the tree-based models, peak-hour RMSE is substantially higher than off-peak RMSE. This indicates that good overall average performance does not mean that the models predict the busiest hours equally well.

The notebooks contain additional diagnostics, including fold-level variation, negative predictions, unusually large errors, and performance across weather conditions. These analyses are descriptive; they do not establish causal explanations for demand changes.

## 7. Repository Structure

```text
ml-2026-MiniProject/
├── data/
│   └── hour.csv
├── figures/                         # Generated plots
├── results/                         # Fold metrics and out-of-fold predictions
├── src/
│   ├── data.py                      # Shared data loading and preprocessing
│   └── validation.py                # Splitters, metrics, and evaluation routine
├── 01_eda.ipynb                     # Exploratory data analysis
├── 02_baseline.ipynb                # Baseline models
├── 03_ridge.ipynb                   # Ridge Regression experiments
├── 04_random_forest.ipynb           # Random Forest experiments
├── 05_gradient_boosting.ipynb       # HistGradientBoosting experiments
├── 06_comparison_and_errors.ipynb   # Global comparison and error diagnostics
├── requirements.txt                 # Python dependencies
└── README.md
```

## 8. Reproducing the Experiments

Run the notebooks from the repository root so that imports such as `from src.data import load_data` resolve correctly.

### Environment setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Notebook execution order

1. `01_eda.ipynb`
2. `02_baseline.ipynb`
3. `03_ridge.ipynb`
4. `04_random_forest.ipynb`
5. `05_gradient_boosting.ipynb`
6. `06_comparison_and_errors.ipynb`

The model notebooks save fold metrics and out-of-fold predictions in `results/`. The final comparison notebook reads these saved files rather than retraining the models. Before running it, ensure that each of the seven model variants (the two baselines and five candidate variants) has generated its `*_folds.csv` and `*_preds.csv` files for all three validation schemes.

The figures can be recreated by executing the corresponding notebook cells.

## 9. Limitations

- The dataset covers only two years and one city.
- The main experiments exclude the year indicator and do not use lagged rental demand. Models therefore cannot directly use recent demand history or the explicit year label.
- Chronological validation evaluates later periods within the observed dataset; it does not guarantee performance under future structural changes.
- Random and blocked validation are diagnostic comparisons, not realistic simulations of predicting only future observations.
- The selected hyperparameters are limited to the grids explored in the notebooks.
- The outer cross-validation results are also used to compare the candidate model families. We do not have an additional, untouched final test set for a final independent estimate after choosing among all model families.
- Error patterns and feature-importance analyses are descriptive and do not establish causal explanations.

## 10. AI-Use Disclosure

Complete this section jointly before submission. The course requires an accurate disclosure of which AI tools were used, how they influenced the work, what code or analysis the project authors wrote or substantially changed, and what they independently verified and understood.

Document only what actually occurred, including relevant use for conceptual explanations, code drafting, debugging, methodological review, interpretation, or editing. Distinguish AI-assisted work from work written or substantially modified by the authors, and describe which outputs and assumptions were independently checked.

## 11. References

- [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- Course project specification: Machine Learning, National University of Kyiv-Mohyla Academy.
