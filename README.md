# 23CE051 – RSA External Practical (Set B)

**Subject:** Responsible and Safe AI (AIML4001) · CSPIT, CHARUSAT
**Student:** Darshan Kachhiya (23CE051) · **Date:** 25/11/2026

## Explaining Instant EMI-Finance Decisions at an Electronics Showroom
An explainability pipeline for an NBFC's instant EMI-loan approval model, using the German Credit dataset (UCI Statlog, 1000 applicants, 20 attributes).

| Task | What is done |
|---|---|
| 1 | Load data, OrdinalEncoder for categorical columns, stratified 80:20 split (random_state = 42) |
| 2 | Random Forest (200 trees, max_depth 8): accuracy and ROC-AUC |
| 3 | SHAP TreeExplainer global importance and summary plots, top 5 features |
| 4 | SHAP waterfall/force plot for a rejected applicant, top 3 reasons |
| 5 | LIME (num_features = 5) for the same applicant, compared with SHAP |
| 6 | Plain-language rejection message and sensitive-attribute check |

## Results
| Metric | Value |
|---|---|
| Test accuracy | 0.77 |
| ROC-AUC | 0.83 |
| Top 5 features | checking_status, duration, credit_amount, savings_status, purpose |
| Rejected applicant | test index 23, P(bad) = 0.57 |
| SHAP top 3 | checking_status=<0, duration=24, personal_status=female div/dep/mar |
| SHAP vs LIME top-3 agreement | 2 / 3 |
| Sensitive attributes in reasons | personal_status, age → REVIEW |

## ✅ Verified on Kaggle — results are identical
The notebook was re-run on Kaggle (`RSA_SetB_Kaggle_executed.ipynb`, dataset at
`/kaggle/input/datasets/darshankachhiya/german-credit/german_credit.csv`).
Every value matches the local run and the PDF report exactly:

| Output | Local run | Kaggle run | Same? |
|---|---|---|---|
| Test accuracy / ROC-AUC | 0.77 / 0.83 | 0.77 / 0.83 | ✅ |
| Confusion matrix [[TN FP][FN TP]] | [[132 8][39 21]] | [[132 8][39 21]] | ✅ |
| Top 5 mean \|SHAP\| | 0.092, 0.042, 0.027, 0.027, 0.025 | 0.092, 0.042, 0.027, 0.027, 0.025 | ✅ |
| Rejected applicant | index 23, P(bad) = 0.568 | index 23, P(bad) = 0.568 | ✅ |
| SHAP top 3 | +0.110, +0.056, +0.038 | +0.110, +0.056, +0.038 | ✅ |
| LIME top 5 | +0.123, −0.056, +0.036, +0.036, +0.033 | +0.123, −0.056, +0.036, +0.036, +0.033 | ✅ |
| SHAP vs LIME agreement | 2 / 3 | 2 / 3 | ✅ |
| Sensitive attributes | personal_status, age → REVIEW | personal_status, age → REVIEW | ✅ |
| Model without sensitive attributes | acc 0.79, AUC 0.81, applicant P(bad) 0.59 (still rejected) | same | ✅ |

The results can be reproduced because `random_state = 42` is used for the split, the Random Forest and LIME.

## Files
| File | Description |
|---|---|
| `rsa_setB_explainability.py` | Full solution script |
| `rsa_setB_explainability.ipynb` | Notebook with outputs |
| `RSA_SetB_Kaggle.ipynb` | Kaggle-ready notebook (auto-loads dataset from `/kaggle/input`) |
| `RSA_SetB_Kaggle_executed.ipynb` | The notebook as run on Kaggle, with all outputs |
| `23CE051_External_Practical_Exam.pdf` | Practical report (aim, algorithm, program, output, Q1–Q4, conclusion) |
| `german_credit.csv` | Dataset: UCI Statlog German Credit (1000 rows, 20 attributes + class), decoded to readable labels |
| `make_csv.py` | Converts the original coded UCI `german.data` (A11, A12 ...) into `german_credit.csv` |
| `plots/` | shap_summary, shap_summary_bar, shap_waterfall, shap_force, lime_explanation |
| `run_output.txt` | Console output of the script |

## How to run
```bash
pip install pandas scikit-learn shap lime matplotlib
python rsa_setB_explainability.py   # german_credit.csv must be in the same folder
```
