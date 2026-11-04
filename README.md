# Bike-Sharing Demand Prediction — Machine Learning Mini-Project

**Course:** Machine Learning, National University of Kyiv-Mohyla Academy  
**Project type:** Empirical regression study  
**Target:** Hourly bike-sharing demand (`cnt`)

## 1. Project Overview

This project investigates how well different machine-learning approaches predict the number of bike rentals per hour using the UCI Bike Sharing dataset.

The goal is not simply to obtain the lowest prediction error. It is to conduct a reproducible empirical investigation, justify the modelling and validation choices, compare the resulting experiments, and analyse where and why the models make errors.

A central question of the project is:

> How do estimates of predictive performance change when hourly bike-sharing data are evaluated with random, blocked, and chronological validation, and does this difference depend on the model family?

The project also examines whether a logarithmic target transformation or a Poisson loss changes the prediction-error profile.

## 2. Dataset

The project uses the **hourly** file (`data/hour.csv`) from the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). It contains hourly rental counts in Washington, D.C., for 2011–2012, together with calendar and weather-related features.

- **Target:** `cnt`, the total number of bike rentals in an hour.
- **Observations:** 17,379 recorded hours.
- **Time coverage:** 2011–2012.
- **Prediction task:** regression of hourly rental counts.

The preprocessing and feature-selection decisions are documented in `01_eda.ipynb` and implemented centrally in `src/data.py`.

In particular, variables that directly reveal the target (`casual` and `registered`) are excluded to avoid target leakage. The timestamp is used as the index, and the observations are sorted chronologically. The year indicator is excluded from the main modelling experiments so that the models do not directly use the 2011/2012 label.

The dataset contains some missing timestamps and 22 zero values in `hum`, which are treated as missing by the shared data loader. Missing timestamps are not silently filled with synthetic rental counts.

## 3. Repository Structure

```text
ml-2026-MiniProject/
├── data/
│   └── hour.csv
├── figures/                         # Generated plots
├── results/                         # Saved fold metrics and out-of-fold predictions
├── src/
│   ├── data.py                      # Shared data loading and preprocessing
│   └── validation.py                # Shared splitters, metrics, and evaluation routine
├── 01_eda.ipynb                     # Exploratory data analysis
├── 02_baseline.ipynb                 # Baseline models
├── 03_ridge.ipynb                    # Ridge Regression experiments
├── 04_random_forest.ipynb            # Random Forest experiments
├── 05_gradient_boosting.ipynb         # HistGradientBoosting experiments
├── 06_comparison_and_errors.ipynb     # Global comparison and error diagnostics
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

## 4. Experimental Design

### Baseline Models

Two simple reference models are used to establish a minimum level of predictive performance:

1. **Constant baseline** — predicts the mean rental count observed in the training fold.
2. **Hour × day-type baseline** — predicts the training-fold mean for the corresponding hour and day type, falling back to the training mean for unseen combinations.

All baseline predictions are generated within the validation procedure so that test-fold information is not used to calculate the baseline predictions.

### Candidate Models

The project investigates three model families:

| Model family | Experiment variants | Purpose |
|---|---|---|
| Ridge Regression | `ridge`, `ridge_log` | Regularised linear regression; the second variant uses a `log1p` transformation of the target. |
| Random Forest | `rf` | Bagged decision trees that can capture nonlinear relationships and feature interactions. |
| Histogram-based Gradient Boosting | `hgb`, `hgb_poisson` | Sequential tree boosting; the variants use squared-error and Poisson losses, respectively. |

The two Ridge variants and the two Gradient Boosting variants are alternative configurations within the same model family, not additional model families.

### Preprocessing

Preprocessing is model-specific and is fitted within the training portion of each cross-validation fold:

- **Ridge Regression:** uses a preprocessing pipeline with categorical encoding, numerical imputation and scaling, a quadratic temperature feature, and an hour-by-working-day interaction. The target-transformed variant applies `log1p` during fitting and transforms predictions back with `expm1`.
- **Random Forest:** uses tree-based modelling without feature scaling or one-hot encoding; missing humidity values are imputed within the pipeline.
- **HistGradientBoosting:** uses native handling of missing values and categorical features. Automatic early stopping is disabled so that an additional random internal validation split is not introduced into the time-ordered experiment.

The exact transformations and parameter grids are documented in the corresponding model notebooks.

## 5. Validation Strategy

The shared implementation in `src/validation.py` provides three validation schemes:

1. **Random K-Fold:** shuffled folds. Training data can contain observations later in time than observations in the test fold.
2. **Blocked K-Fold:** contiguous test folds without shuffling. The training portion includes observations on both sides of a test block, so this is not a simulation of forecasting future observations.
3. **Chronological expanding-window validation:** the initial training period covers the first 50% of the time span. The remaining time span is divided into five consecutive test periods of equal calendar duration. Each fold is trained only on observations earlier than its test period.

For model experiments with hyperparameter tuning, an **outer evaluation loop** estimates performance and an **inner cross-validation loop** selects hyperparameters using only the outer training data. The outer test fold is not used during model selection. The outer loop uses five folds; the inner search uses three splits corresponding to the selected validation scheme.

The chronological scheme is the primary evaluation setting for the intended future-demand prediction task. Random and blocked validation are retained to study how the validation design affects the estimated performance.

The common evaluation code also checks feature/target index alignment and saves fold-level metrics and out-of-fold predictions. The random seed is set to `42` wherever applicable.

## 6. Evaluation Metrics

The experiments use four regression metrics:

- **RMSE (Root Mean Squared Error):** measured in rentals per hour and penalises large errors more strongly. It is used as the primary hyperparameter-search metric.
- **MAE (Mean Absolute Error):** average absolute prediction error, also measured in rentals per hour.
- **R²:** compares the model with a mean-prediction reference in terms of explained variance.
- **RMSLE (Root Mean Squared Logarithmic Error):** evaluates errors on a logarithmic scale and is useful for examining relative errors across different demand levels.

Negative predictions are retained for RMSE, MAE, and R². For RMSLE only, negative predictions are clipped to zero before applying the logarithm; negative-prediction rates are therefore analysed separately.

## 7. Results and Comparison

The model-specific notebooks report each candidate's own fold-level results, selected hyperparameters, and error diagnostics. The global comparison notebook combines the saved results from all model families and baseline models.

The global analysis is designed to include:

- fold-averaged performance under each validation scheme;
- a matched evaluation on the same chronological test timestamps (the 2012 observations evaluated by chronological validation);
- the difference between random and chronological performance estimates;
- variation in performance across chronological test periods;
- residual and bias diagnostics, including peak versus off-peak demand.

**Final quantitative results and model selection should be added here after the complete, clean execution of all notebooks.** Results from different validation schemes should not be interpreted as if they necessarily used identical training sets. In particular, random and blocked validation may use future observations relative to a test timestamp, unlike chronological validation.

## 8. Reproducing the Experiments

Run the notebooks from the repository root so that imports such as `from src.data import load_data` resolve correctly.

### Environment Setup

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

### Run Order

Execute the notebooks in this order:

1. `01_eda.ipynb`
2. `02_baseline.ipynb`
3. `03_ridge.ipynb`
4. `04_random_forest.ipynb`
5. `05_gradient_boosting.ipynb`
6. `06_comparison_and_errors.ipynb`

The model notebooks save fold metrics and out-of-fold predictions in `results/`. The final comparison notebook reads these saved files and does not need to retrain the models.

Before running the global comparison, ensure that every model has generated both `*_folds.csv` and `*_preds.csv` files for all three validation schemes (`random`, `blocked`, and `chrono`).

The `figures/` directory contains plots generated by the notebooks. These figures can be recreated by executing the relevant notebook cells.

## 9. Limitations

The findings are limited by the available data and experimental design:

- The dataset covers only two years and one city.
- The main experiments exclude the year indicator and do not use lagged rental demand, so they cannot directly use recent demand history or explicitly encode the year-to-year change.
- Chronological validation evaluates future periods within the observed dataset; it does not establish performance under arbitrary future structural changes.
- Random and blocked validation are diagnostic comparisons, not realistic simulations of forecasting future observations.
- The selected hyperparameters are limited to the grids explored in the model notebooks.
- Error patterns and feature-importance analyses are descriptive and do not establish causal explanations for demand changes.

## 10. AI-Use Disclosure

> **Complete this section jointly before submission.** The course requires a disclosure of which AI tools were used, how they influenced the work, what code or analysis the project authors wrote or substantially changed, and what they independently verified and understood.

Suggested points to document accurately:

- Tools used and the tasks for which they were used (for example, conceptual explanations, code drafting, debugging, methodological review, interpretation of results, or editing documentation).
- Whether AI suggestions influenced preprocessing, validation, model selection, metrics, or interpretation decisions.
- Which code cells and analyses were written by the project authors, generated with assistance, or substantially modified by the authors.
- Which outputs, assumptions, calculations, and code paths were independently checked by the project authors.

The authors should only retain statements that accurately describe their actual use of AI tools and their own contributions.

## 11. References

- [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- Course project specification: Machine Learning, National University of Kyiv-Mohyla Academy.
