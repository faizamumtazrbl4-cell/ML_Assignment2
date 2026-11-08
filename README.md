# PCOS Prediction: Classification Models and Streamlit App

M.Tech (AIML) · Machine Learning · Assignment 2

**Live Streamlit app:** `<PASTE STREAMLIT COMMUNITY CLOUD LINK HERE>`

---

## a. Problem statement

Polycystic ovary syndrome (PCOS) is a common hormonal disorder in women. It is frequently diagnosed late because its signs (irregular periods, acne, excess hair growth, weight gain) overlap with many other conditions, and confirmation needs blood tests and an ultrasound.

The goal of this project is to build and compare **five machine-learning classifiers** that predict, from a patient's clinical, hormonal and ultrasound measurements, whether she has PCOS (binary classification: `0 = No PCOS`, `1 = PCOS`). The models are evaluated on a held-out test set with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC) and are deployed in an interactive Streamlit app, which also includes a gentle self-check page for women.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | Kaggle, *Polycystic Ovary Syndrome (PCOS)* clinical dataset (public, `PCOS_data.csv`) |
| Task | Binary classification, target column `pcos` |
| Instances | **541** patients (minimum required: 500) |
| Features | **41** predictors (minimum required: 12), after dropping 2 ID columns from the original 44 columns |
| Class balance | 364 No PCOS (67.3 %) and 177 PCOS (32.7 %) |
| Missing data | 4 cells missing or unparseable in the raw file, and 6 physiologically impossible readings, all repaired (no rows dropped) |

**Feature groups**
- **Demographics and body:** age, weight, height, BMI, blood group, waist, hip, waist:hip ratio
- **Vitals:** pulse rate, respiratory rate, haemoglobin, blood pressure (systolic and diastolic)
- **Menstrual and reproductive history:** cycle regularity, cycle length, years married, pregnancy status, number of abortions
- **Hormone and lab tests:** FSH, LH, FSH/LH ratio, TSH, AMH, prolactin, vitamin D3, progesterone, random blood sugar, β-hCG (two readings)
- **Symptoms and lifestyle (0/1):** weight gain, hair growth, skin darkening, hair loss, pimples, fast food, regular exercise
- **Ultrasound:** follicle count (left and right ovary), average follicle size (left and right), endometrium thickness

**Preparation (see `notebooks/`)**
1. Cleaned the headers to `snake_case`, fixed two numeric columns stored as text (`'1.99.'` repaired, `'a'` set to missing), and dropped the ID columns.
2. Set six impossible readings to missing and imputed them (pulse 13 and 18 bpm, BP 12/8 mmHg, FSH 5052 and LH 2018 mIU/mL).
3. Imputed the missing cells with the median (continuous columns) or mode (binary flag).
4. Stratified 80/20 split with `random_state=26`: **432 train / 109 test** rows. PCOS share is 32.6 % in train and 33.0 % in test.
5. A preprocessing pipeline fitted on the **training set only**: median impute, `log1p` on 14 skewed hormone and lab columns, standardisation, and one-hot encoding of `blood_group` and `cycle_ri`.

`test_data.csv` in the repository root is the held-out test set (109 rows, all 41 features plus the `pcos` target) used by the Streamlit app.

## c. GitHub repository link

**Repository:** `<PASTE GITHUB REPOSITORY LINK HERE>`

```
project-folder/
│-- app.py                      Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv               held-out test data (109 rows) to upload in the app
│-- model/                      saved trained models, one full pipeline per classifier
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- svm.joblib
│   │-- preprocessor.joblib     fitted preprocessing artifact
│   │-- feature_config.joblib   feature order and column groups
│   └-- feature_defaults.json   typical values used by the check-in form
│-- notebooks/                  source code (Jupyter)
│   │-- 01_data_audit_and_cleaning.ipynb
│   │-- 02_eda.ipynb
│   │-- 03_train_data_prep.ipynb
│   └-- 04_modelling.ipynb
│-- src/                        source code (Python modules)
│   │-- data_utils.py
│   │-- app_logic.py
│   └-- make_defaults.py
│-- data/raw/PCOS_data.csv      original dataset
│-- data/processed/             cleaned data and training split
└-- reports/                    metrics tables and figures
```

**Run locally**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Each saved model is a complete scikit-learn `Pipeline` (preprocessing followed by the classifier). The app loads these files, computes the metrics on the uploaded test CSV, and **does not retrain** anything.

## d. Models used

All five models were trained on the same training split and tuned with `GridSearchCV` (stratified 5-fold CV on the training set only, scoring on ROC-AUC). Results below are on the **held-out test set (109 patients)**.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.9174** | 0.9536 | 0.8649 | 0.8889 | **0.8767** | **0.8148** |
| Decision Tree | 0.8257 | 0.8579 | 0.7179 | 0.7778 | 0.7467 | 0.6153 |
| KNN | 0.8624 | **0.9547** | **0.9565** | 0.6111 | 0.7458 | 0.6886 |
| Naive Bayes | 0.8532 | 0.9433 | 0.7500 | 0.8333 | 0.7895 | 0.6795 |
| Support Vector Machine | 0.8257 | 0.9384 | 0.6809 | **0.8889** | 0.7711 | 0.6490 |

Best value in each column is in bold. Precision, Recall and F1 are for the positive class (PCOS).

**Tuned hyper-parameters**

| Model | Best settings | CV AUC (default → tuned) |
|---|---|---|
| Logistic Regression | L1 penalty, C = 0.5 | 0.928 → 0.940 |
| Decision Tree | Gini, max depth 4, min samples per leaf 10 | 0.844 → 0.896 |
| KNN | 21 neighbours, distance weighting, Manhattan distance | 0.857 → 0.930 |
| Naive Bayes (Gaussian) | var_smoothing = 0.1 | 0.901 → 0.931 |
| Support Vector Machine | RBF kernel, C = 0.5, gamma = 0.01, balanced class weights | 0.933 → 0.939 |

**Confusion matrices on the test set** (73 No PCOS, 36 PCOS)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 68 | 5 | 4 | 32 |
| Decision Tree | 62 | 11 | 8 | 28 |
| KNN | 72 | 1 | 14 | 22 |
| Naive Bayes | 63 | 10 | 6 | 30 |
| Support Vector Machine | 58 | 15 | 4 | 32 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best all-round model: highest accuracy (91.7 %), F1 (0.877) and MCC (0.815), with a balanced precision (0.865) and recall (0.889). It missed only 4 of 36 PCOS patients. Train and test accuracy differ by under 0.5 points, so it generalises very well. The dataset's strong, mostly monotonic signals (follicle counts, cycle irregularity, hair growth, skin darkening, weight gain) suit a linear model, and L1 regularisation helps with the many correlated features. |
| Decision Tree | Weakest model on almost every metric (accuracy 82.6 %, AUC 0.858, MCC 0.615). Tuning helped (CV AUC 0.844 to 0.896) but a single tree is high-variance on only 432 training rows, and its test accuracy is about 8 points below its training accuracy. Easy to interpret, but not competitive here. |
| KNN | Highest AUC (0.955) and highest precision (0.957, only 1 false positive), so it ranks patients well. However recall is just 0.611: it misses 14 of 36 PCOS patients at the default 0.5 threshold, which is the worst result for a screening use case. Training accuracy is 1.00 against 0.86 on test, a sign of overfitting. Distance-based methods also suffer from the 50-column feature space. |
| Naive Bayes | A solid mid-table model (accuracy 85.3 %, recall 0.833, AUC 0.943) that is fast and needs very little tuning. Its independence assumption is violated by correlated features (BMI and weight, FSH/LH and FSH, left and right follicle counts), which holds back precision (0.750) and MCC (0.680). |
| Support Vector Machine | Ties Logistic Regression on recall (0.889, 4 missed cases) and has a strong AUC (0.938), but its precision is low (0.681): 15 healthy women were flagged as PCOS. Balanced class weights push it toward over-predicting PCOS, so accuracy (82.6 %) and MCC (0.649) fall behind. It also shows a visible train-to-test gap. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It ranks first on accuracy, F1 and MCC, ties for the best recall, and is within 0.002 of the best AUC. It also has the smallest train-to-test gap and ranked first in a repeated 5-fold cross-validation on the full data (mean AUC 0.945). It is also the simplest and most interpretable model. KNN has the top AUC and precision but misses too many PCOS cases to be a good screening model. Note that the test set is small (109 rows), so the AUC differences between the top four models are within noise. |

## Streamlit app features

- **Dataset upload (CSV):** upload test data containing all feature columns plus the `pcos` target. A bundled sample is available. Missing columns, a bad target and blank cells are reported clearly.
- **Model selection dropdown:** choose any of the five saved models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC are shown for the selected model, and again for all five models side by side.
- **Confusion matrix and classification report:** shown for the selected model, along with a ROC curve and a download button for the predictions.
- **My check-in tab:** a private, plain-language form where a woman enters her own details (cycle, symptoms, optional lab and scan results) and gets an estimate with the main reasons behind it. This is an awareness tool, not a diagnosis.

## Limitations

- The data comes from a single source and is small (541 rows), so results may not generalise to other populations.
- Ultrasound follicle counts are part of the clinical criteria for PCOS and are therefore highly predictive. This is a form of near-leakage, so scores on this dataset are optimistic compared with a setting where scan data is unavailable.
- With a 109-row test set, one extra mistake moves a metric by 1 to 2 points.
- Not a medical device and not a substitute for a doctor.
