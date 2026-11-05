# AI Use Disclosure

## 1. Tools Used

| Tool | Version / Model | Who Used It |
|---|---|---|
| Claude | Sonnet 5.5 | Polina, Slava |
| ChatGPT | GPT-6 | Polina |
| Gemini | Flash, Pro | Slava |

## 2. How the Tools Were Used

- [x] **Research and problem formulation:** Exploring possible research directions, brainstorming interesting research questions, and refining the project's focus on the impact of validation strategies on model performance estimates.
- [x] **Methodological discussions:** Explaining and discussing validation schemes, nested cross-validation, data leakage, preprocessing pipelines, and evaluation metrics.
- [x] **Code writing:** Drafting and adapting selected code snippets for `src/data.py`, `src/validation.py`, and the experimental notebooks (`01_eda.ipynb`–`06_comparison_and_errors.ipynb`).
- [x] **Debugging:** Identifying and resolving programming issues related to data processing, model pipelines, cross-validation, and result visualisation.
- [x] **Interpretation of results:** Assistance with comparing model performance, analysing differences between validation strategies, interpreting error patterns, and formulating conclusions.
- [x] **Editing the report / README:** Improving the structure, clarity, and consistency of the project documentation and written explanations.

## 3. How AI Influenced Methodological Decisions

| Decision | AI's Influence | What We Did Independently |
|---|---|---|
| Choice of validation schemes (random, blocked, and chronological) | Helped explain the differences between the schemes and suggested using blocked CV as an intermediate diagnostic comparison. | Evaluated the relevance of each scheme to our research question and retained all three for the experiments. |
| Using chronological validation in the inner loop as well | Explained why the inner validation should follow the same temporal logic as the outer evaluation to avoid selecting hyperparameters using future observations. | Implemented the nested validation procedure and checked that the inner search used only the corresponding outer training data. |
| Choice of models (Ridge, Random Forest, Gradient Boosting) | Helped explore and compare model families with different characteristics, including linear and nonlinear approaches. | Selected and implemented the candidate models, defined their configurations, and conducted the experiments. |
| Choice of evaluation metrics | Explained the differences between RMSE, MAE, R², and RMSLE and the types of errors they emphasise. | Used these metrics to evaluate the models and selected RMSE as the primary metric for hyperparameter tuning. |
| Code structure (`data.py`, `validation.py`) | Suggested ways to organise reusable preprocessing, validation, and evaluation logic separately from model-specific experiments. | Adapted the structure to the project workflow and integrated the shared functions into the notebooks. |

**Corrections and disagreements:** AI-assisted discussions also helped us identify potentially misleading interpretations. For example, we clarified that blocked cross-validation is not a realistic simulation of forecasting future observations because its training data can include later time periods. We also distinguished optimistic performance estimates under random validation from direct target leakage. These distinctions were reflected in the final methodology and interpretation of results.

## 4. Representative Prompts

1. **Prompt:** «Як інтерпретувати суттєву різницю між результатами Random та Chronological CV у нашому експерименті? Якщо ми не використовуємо лаги цільової змінної `cnt`, то які саме часові залежності або особливості датасету можуть пояснювати таку різницю? Чи коректно називати це data leakage?»

   **How we used the response:** We used AI to critically examine the possible reasons behind the performance gap and distinguish temporal dependence and optimistic evaluation from direct target leakage. This helped us formulate a more accurate interpretation of the results and avoid overstating what our experiment demonstrates.

2. **Prompt:** «Яку додаткову дослідницьку цінність дає Blocked CV порівняно з Random та Chronological CV? Чи дозволяє ця третя схема краще зрозуміти, за рахунок чого виникає різниця в оцінках якості моделей?»

   **How we used the response:** We discussed whether including a third validation strategy was justified by our research question. We retained blocked cross-validation as an intermediate diagnostic comparison and used the results to examine how performance estimates change when test observations are grouped into contiguous time periods while future observations are still available for training.

3. **Prompt:** «Як правильно порівнювати результати різних схем валідації, якщо ми оцінюємо моделі на однакових часових спостереженнях 2012 року, але тренувальні вибірки в Random, Blocked та Chronological CV відрізняються? Які висновки дозволяє зробити таке порівняння, а які — ні?»

   **How we used the response:** We used AI to examine the interpretation and limitations of our matched evaluation. This helped us distinguish between using the same test timestamps and using the same training data. We retained this comparison because it makes the evaluated observations consistent across schemes, while explicitly acknowledging that the training information differs.

4. **Prompt:** «Що саме оцінює Nested Cross-Validation у нашому експерименті? Чи можна вважати результати зовнішніх фолдів незалежною фінальною оцінкою після того, як ми використовуємо їх для порівняння та вибору моделей?»

   **How we used the response:** We used AI to examine what our nested validation procedure actually estimates and identify the limitations of the experimental design. This helped us distinguish hyperparameter selection from outer-fold evaluation and recognise that, although the outer test folds are excluded from the corresponding inner searches, we do not have an additional untouched test set after comparing all candidate models.
## 5. Work We Wrote or Substantially Modified Ourselves

| Part of the Project | Author(s) | What We Did Independently |
|---|---|---|
| `src/data.py` | Slava | Implemented the shared data-loading and preprocessing module. Handled timestamp construction and chronological sorting, selected the features used in the experiments, excluded target-derived variables to prevent data leakage, and implemented the treatment of invalid humidity values and weather categories. |
| `src/validation.py` | Slava | Implemented the shared validation and evaluation framework, including random, blocked, and chronological splitters, the evaluation metrics, and the procedure for training and evaluating models across folds. Implemented the saving of fold-level results and out-of-fold predictions for further analysis. |
| `01_eda.ipynb` | Polina | Conducted exploratory data analysis to understand the dataset's structure, data quality, target distribution, temporal patterns, and relationships between features. Investigated missing timestamps, potential data leakage, correlations between variables, differences between working and non-working days, and changes in demand between 2011 and 2012. Used the findings to motivate preprocessing and modelling decisions. |
| `02_baseline.ipynb` | Slava | Implemented and evaluated two baseline approaches: a constant mean predictor and an hour-by-day-type predictor. Defined their prediction logic, including fallback behaviour for unseen feature combinations, and evaluated them under the selected validation schemes to establish reference performance levels. |
| `03_ridge.ipynb` | Polina | Developed the Ridge Regression experiments, including preprocessing pipelines, categorical encoding, feature engineering, and hyperparameter tuning. Compared models trained on the original and log-transformed target. Analysed fold-level performance, prediction behaviour, and model errors. |
| `04_random_forest.ipynb` | Slava | Developed the Random Forest experiment, selected and evaluated model configurations, and performed hyperparameter tuning within the cross-validation procedure. Analysed its predictive performance under different validation schemes and examined its errors and model behaviour. |
| `05_gradient_boosting.ipynb` | Polina | Implemented and compared two HistGradientBoosting variants using squared-error and Poisson losses. Designed the hyperparameter search, evaluated both variants under the validation framework, and investigated their performance differences, residual patterns, negative predictions, and ability to handle high-demand observations. |
| `06_comparison_and_errors.ipynb` | Polina, Slava | Combined the results from all model families and baselines. Checked consistency and alignment of out-of-fold predictions, compared performance across validation schemes, and conducted a matched evaluation on the same chronological test timestamps. Analysed performance differences between models, prediction bias, peak and off-peak errors, weather-related patterns, and unusually large errors. Jointly interpreted the findings and developed the final conclusions. |
| README / Conclusions | Polina | Prepared and revised the project documentation, formulated the central research question, summarised the experimental methodology and quantitative findings, and documented the limitations of the study. Ensured that the conclusions reflected the actual experimental results and the scope of the investigation. |

## 6. What We Independently Checked and Understood

- **Chronological validation:** Checked how the expanding training window is constructed and verified that, in each chronological fold, training observations always precede the test period.
- **Nested cross-validation:** Reviewed how the inner and outer loops interact. Confirmed that hyperparameter tuning is performed only on the outer training data and that the outer test fold remains unseen during model selection.
- **Preprocessing and data leakage:** Examined the feature selection and preprocessing pipelines. Verified why `casual` and `registered` must be excluded, since their sum directly equals the target `cnt`, and reviewed how missing values, scaling, and categorical encoding are handled.
- **Matched evaluation:** Checked that all models and validation schemes are compared on the same 8,722 chronological test timestamps. Reviewed timestamp alignment and consistency of the actual target values across predictions.
- **Results and metrics:** Compared the reported performance metrics with the saved fold-level results and out-of-fold predictions. Reviewed the differences between RMSE, MAE, R², and RMSLE and examined how negative predictions and large errors affect model evaluation.
- **Interpretation and limitations:** Discussed why random validation can produce optimistic performance estimates and why blocked validation does not fully simulate forecasting future observations. Also identified the limitation of not having a separate untouched test set after comparing the candidate models.

## 7. Limitations and Acknowledgements

- **AI-generated inaccuracies:** Some initial explanations did not fully reflect our actual implementation. For example, when discussing why random validation may produce optimistic results, AI initially used previous-hour demand as an illustrative example. This could incorrectly suggest that our models use lagged `cnt` values. We checked the feature set, clarified that no lagged target features were included, and revised the explanation to focus on temporal patterns, similar observations, and the availability of future-period data in training folds.
- **Critical evaluation of suggestions:** Some explanations initially oversimplified methodological issues. For example, we discussed and refined the interpretation of blocked cross-validation to avoid treating it as an experiment that isolates a single cause of the performance gap. We also had to adapt some code suggestions to match our actual implementation.
- **Remaining limitations:** We cannot fully explain all observed prediction errors or year-to-year changes in demand using the available features. Some interpretations therefore remain hypotheses rather than established explanations.
- **Acknowledgement:** AI assistance was useful for exploring ideas, understanding concepts, and debugging, but its responses were not always accurate or directly applicable. The project authors remained responsible for checking suggestions against the code, dataset, and experimental results.
