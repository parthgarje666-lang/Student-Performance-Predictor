"""
Predicting Student Academic Performance - end-to-end training script.

Based on the ML mini project report by Parth Garje (CS3107).

Usage:
    python src/train.py                              # synthetic demo data
    python src/train.py --data data/student-mat.csv  # real UCI data (sep=';')

Outputs: figs/*.png, results.csv, models/student_model.joblib
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FIGS, MODELS = ROOT / "figs", ROOT / "models"
PASS_THRESHOLD = 10  # G3 >= 10 -> "Pass", else "Needs Support"


# ------------------------------------------------------------------ 1. DATA
def make_synthetic_data(n=395, seed=SEED):
    """Synthetic data with the same schema as UCI student-mat.csv (demo only)."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)
    d = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], n, p=[.88, .12]),
        "sex": rng.choice(["F", "M"], n),
        "age": rng.integers(15, 20, n),
        "address": rng.choice(["U", "R"], n, p=[.78, .22]),
        "famsize": rng.choice(["GT3", "LE3"], n, p=[.71, .29]),
        "Medu": rng.integers(0, 5, n), "Fedu": rng.integers(0, 5, n),
        "studytime": rng.choice([1, 2, 3, 4], n, p=[.27, .50, .16, .07]),
        "failures": rng.choice([0, 1, 2, 3], n, p=[.79, .13, .04, .04]),
        "schoolsup": rng.choice(["yes", "no"], n, p=[.13, .87]),
        "famsup": rng.choice(["yes", "no"], n, p=[.61, .39]),
        "paid": rng.choice(["yes", "no"], n, p=[.46, .54]),
        "activities": rng.choice(["yes", "no"], n),
        "internet": rng.choice(["yes", "no"], n, p=[.83, .17]),
        "romantic": rng.choice(["yes", "no"], n, p=[.33, .67]),
        "famrel": rng.integers(1, 6, n), "freetime": rng.integers(1, 6, n),
        "goout": rng.integers(1, 6, n), "Dalc": rng.integers(1, 6, n),
        "Walc": rng.integers(1, 6, n), "health": rng.integers(1, 6, n),
    })
    d["absences"] = np.clip(rng.poisson(5, n) + (rng.random(n) < .05) * rng.integers(10, 40, n), 0, 75)
    score = (10.4 + 3.2 * ability + 0.6 * (d.studytime - 2) - 1.3 * d.failures
             - 0.06 * d.absences + 0.2 * (d.Medu - 2) - 0.2 * (d.goout - 3))
    d["G1"] = np.clip(np.round(score + rng.normal(0, 1.6, n)), 0, 20).astype(int)
    d["G2"] = np.clip(np.round(score + rng.normal(0, 1.3, n)), 0, 20).astype(int)
    d["G3"] = np.clip(np.round(score + 0.4 * (d.G2 - score) + rng.normal(0, 1.2, n)), 0, 20).astype(int)
    # inject realistic problems: missing values + 4 duplicate rows
    for col, frac in [("absences", .03), ("studytime", .02), ("internet", .03), ("famsize", .02)]:
        d.loc[rng.random(n) < frac, col] = np.nan
    return pd.concat([d, d.sample(4, random_state=seed)], ignore_index=True)


def load_data(path):
    if path:
        print(f"Loading real dataset: {path}")
        return pd.read_csv(path, sep=";")
    print("No --data given -> using SYNTHETIC demo data (results are not real-student results).")
    return make_synthetic_data()


# ------------------------------------------------------------------ 2. PREP
def prepare(df):
    print("Shape:", df.shape)
    miss = df.isna().sum()
    print("Missing values:\n", miss[miss > 0].to_string() if miss.any() else "none")
    print("Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    for c in df.select_dtypes(exclude="number"):
        df[c] = df[c].str.strip()
    df["Result"] = np.where(df["G3"] >= PASS_THRESHOLD, "Pass", "Needs Support")
    print("\nTarget distribution:\n", df["Result"].value_counts().to_string())

    # feature engineering
    df["prior_avg"] = (df["G1"] + df["G2"]) / 2
    df["absence_band"] = pd.cut(df["absences"], [-1, 3, 10, 1000],
                                labels=["low", "medium", "high"]).astype(object)

    X = df.drop(columns=["G3", "Result"])  # no target leakage
    y = (df["Result"] == "Pass").astype(int)
    cat = X.select_dtypes(exclude="number").columns
    X[cat] = X[cat].astype(object).where(X[cat].notna(), np.nan)
    return df, X, y


def make_preprocessor(X, scale):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])


def evaluate(model, X_, y_):
    p, proba = model.predict(X_), model.predict_proba(X_)[:, 1]
    return dict(Accuracy=accuracy_score(y_, p), Precision=precision_score(y_, p, zero_division=0),
                Recall=recall_score(y_, p), F1=f1_score(y_, p),
                MacroF1=f1_score(y_, p, average="macro"), ROC_AUC=roc_auc_score(y_, proba))


# ------------------------------------------------------------------ MAIN
def main(data_path):
    FIGS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df, X, y = prepare(load_data(data_path))

    # EDA plot
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    sns.countplot(x="Result", data=df, ax=ax[0], order=["Needs Support", "Pass"],
                  hue="Result", palette=["#d9534f", "#5cb85c"], legend=False)
    ax[0].set_title("Target distribution")
    sns.histplot(df["G3"], bins=21, kde=True, ax=ax[1], color="#337ab7")
    ax[1].axvline(PASS_THRESHOLD, color="k", ls="--")
    ax[1].set_title(f"Final grade G3 (threshold = {PASS_THRESHOLD})")
    sns.boxplot(x="studytime", y="G3", data=df, ax=ax[2], color="#8fbfe0")
    ax[2].set_title("G3 by study time")
    plt.tight_layout(); plt.savefig(FIGS / "eda.png", dpi=160); plt.close()

    # split BEFORE fitting any preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape} (stratified 80/20, seed={SEED})")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {
        "Dummy baseline": Pipeline([("pre", make_preprocessor(X, False)),
                                    ("m", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline([("pre", make_preprocessor(X, True)),
                                         ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Decision Tree": Pipeline([("pre", make_preprocessor(X, False)),
                                   ("m", DecisionTreeClassifier(random_state=SEED))]),
        "Random Forest": Pipeline([("pre", make_preprocessor(X, False)),
                                   ("m", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
    }
    grids = {
        "Decision Tree": {"m__max_depth": [3, 5, 7, None], "m__min_samples_split": [2, 10],
                          "m__min_samples_leaf": [1, 5]},
        "Random Forest": {"m__n_estimators": [100, 300], "m__max_depth": [None, 5, 10],
                          "m__min_samples_split": [2, 5], "m__min_samples_leaf": [1, 2]},
    }

    print("\n=== HYPERPARAMETER TUNING (5-fold CV, scoring=f1_macro, train set only) ===")
    fitted = {}
    for name, pipe in models.items():
        if name in grids:
            gs = GridSearchCV(pipe, grids[name], cv=cv, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
            fitted[name] = gs.best_estimator_
            print(f"{name}: best CV macro-F1 = {gs.best_score_:.4f}\n  params = {gs.best_params_}")
        else:
            fitted[name] = pipe.fit(X_train, y_train)

    print("\n=== HELD-OUT TEST RESULTS (positive class = Pass) ===")
    res = pd.DataFrame({n: evaluate(m, X_test, y_test) for n, m in fitted.items()}).T.round(4)
    print(res.to_string())
    res.to_csv(ROOT / "results.csv")

    print("\n5-fold CV accuracy on training set (mean +/- std):")
    for n, m in fitted.items():
        s = cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"  {n:<20s} {s.mean():.4f} +/- {s.std():.4f}")
    print("\nBest model by macro-F1:", res.drop("Dummy baseline")["MacroF1"].idxmax())

    rf = fitted["Random Forest"]
    print("\nClassification report - Random Forest:\n",
          classification_report(y_test, rf.predict(X_test),
                                target_names=["Needs Support", "Pass"], digits=3))
    print("Confusion matrix [Needs Support, Pass]:\n", confusion_matrix(y_test, rf.predict(X_test)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, n in zip(ax, ["Decision Tree", "Random Forest"]):
        ConfusionMatrixDisplay.from_estimator(fitted[n], X_test, y_test,
                                              display_labels=["Needs Support", "Pass"],
                                              cmap="Blues", ax=a, colorbar=False)
        a.set_title(f"{n} - confusion matrix")
    plt.tight_layout(); plt.savefig(FIGS / "confusion.png", dpi=160); plt.close()

    pi = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=SEED, scoring="f1_macro")
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    print("\nTop-8 permutation importances (drop in macro-F1):\n", imp.head(8).round(4).to_string())
    plt.figure(figsize=(7, 4)); imp.head(10)[::-1].plot.barh(color="#337ab7")
    plt.xlabel("Mean drop in macro-F1 when shuffled"); plt.title("Random Forest - permutation importance")
    plt.tight_layout(); plt.savefig(FIGS / "importance.png", dpi=160); plt.close()

    # Ablation: no prior grades (early-prediction scenario)
    keep = [c for c in X.columns if c not in ("G1", "G2", "prior_avg")]
    rf2 = Pipeline([("pre", make_preprocessor(X[keep], False)),
                    ("m", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
                    ]).fit(X_train[keep], y_train)
    print("\nAblation - RF WITHOUT G1/G2/prior_avg:",
          {k: round(v, 4) for k, v in evaluate(rf2, X_test[keep], y_test).items()})

    # Fairness check
    pred = rf.predict(X_test)
    print("\nError rate by group (Random Forest, test set):")
    for g in [c for c in ["sex", "address"] if c in X_test]:
        t = (pd.DataFrame({"g": X_test[g].values, "err": pred != y_test.values})
             .groupby("g")["err"].agg(["mean", "size"]).round(3))
        print(f"  {g}:\n{t.to_string()}")

    # Save pipeline + metadata used by the Streamlit app
    defaults = {c: (X[c].median() if X[c].dtype.kind in "iuf" else X[c].mode().iloc[0]) for c in X.columns}
    joblib.dump({"pipeline": rf, "columns": X.columns.tolist(), "defaults": defaults},
                MODELS / "student_model.joblib")
    print("\nModel saved -> models/student_model.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to UCI student-mat.csv (semicolon-separated)")
    main(ap.parse_args().data)
