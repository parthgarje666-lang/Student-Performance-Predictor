# Predicting Student Academic Performance Using Machine Learning

End-to-end ML mini project: data cleaning, feature engineering, model comparison
(Dummy baseline, Logistic Regression, Decision Tree, Random Forest), evaluation,
and a Streamlit deployment app.

Based on the ML mini project report by Parth Garje (CS3107), Faculty: Dr. Prema B. Sahane.

## Problem statement
Predict whether a student will **Pass** (final grade G3 >= 10) or **Needs Support**
from study-related and demographic features. This is a decision-support tool and must
not be used to label students permanently.

## Dataset
UCI Student Performance dataset (Cortez & Silva, 2008):
https://archive.ics.uci.edu/dataset/320/student+performance

If no dataset file is supplied, `src/train.py` generates a **synthetic** dataset with the
same schema so the pipeline can run anywhere. Synthetic results only prove the pipeline
works, not real-student performance. For real results, download `student-mat.csv` into `data/`.

## Project structure
```
.
├── app.py               # Streamlit app
├── src/train.py         # preprocessing, tuning, evaluation, model saving
├── data/                # put student-mat.csv here
├── figs/                # generated plots
├── models/              # saved pipeline (joblib)
├── results.csv          # held-out test metrics
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/train.py                               # synthetic demo data
python src/train.py --data data/student-mat.csv   # real UCI data
streamlit run app.py
```

## Methodology
- Stratified 80/20 split, `random_state=42`; all preprocessing fitted on training data only
- Median / most-frequent imputation, one-hot encoding, scaling only for Logistic Regression
- Engineered features: `prior_avg`, `absence_band`; G3 removed from inputs (no leakage)
- GridSearchCV, 5-fold stratified CV, scoring = macro-F1, training set only
- Metrics: accuracy, precision, recall, F1, macro-F1, ROC-AUC, confusion matrix, permutation importance
- Extras: ablation without prior grades, error-rate check by sex and address

## Results (synthetic demo data, test n = 79)
| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Dummy baseline | 0.519 | 0.519 | 1.000 | 0.683 | 0.342 | 0.500 |
| Decision Tree | 0.810 | 0.861 | 0.756 | 0.805 | 0.810 | 0.884 |
| Random Forest | 0.848 | 0.892 | 0.805 | 0.846 | 0.848 | 0.940 |
| Logistic Regression | 0.848 | 0.837 | 0.878 | 0.857 | 0.848 | 0.944 |

Replace with your numbers after running on the real dataset (see `results.csv`).

## Limitations
Small test set, important factors not in the data, feature importance is not causation,
and prior grades dominate predictions (accuracy drops to about 63% without them).

## References
1. Cortez & Silva (2008). Using Data Mining to Predict Secondary School Student Performance. FUBUTEC.
2. UCI ML Repository, Student Performance dataset.
3. Pedregosa et al. (2011). Scikit-learn. JMLR 12.
4. Breiman (2001). Random Forests. Machine Learning 45.
