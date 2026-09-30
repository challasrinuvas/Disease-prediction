"""
Task 4: Disease Prediction from Medical Data
Objective : Predict the possibility of disease from patient data.
Approach  : Classification on structured medical datasets.
Datasets  : Heart Disease, Diabetes (Pima), Breast Cancer (UCI ML Repository)
Algorithms: Logistic Regression, SVM, Random Forest, XGBoost
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR, OUT_DIR = "data", "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    print(f"[warning] XGBoost could not load ({e}).")
    print("          Using GradientBoosting instead. On Mac, fix with: brew install libomp")
    HAS_XGB = False


# ---------------------------------------------------------------
# 1. DATASET LOADERS  (each returns X: DataFrame, y: Series 0/1)
#    1 = disease present.  Data is cached in ./data after first run.
# ---------------------------------------------------------------
def load_breast_cancer_data():
    d = load_breast_cancer(as_frame=True)
    return d.data, (d.target == 0).astype(int)      # sklearn: 0 = malignant


def load_diabetes_data():
    try:
        d = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
        X, y = d.data.copy(), (d.target.astype(str) == "tested_positive").astype(int)
    except Exception:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class"]
        df = pd.read_csv(url, header=None, names=cols)
        X, y = df.drop(columns="class"), df["class"].astype(int)
    # In this dataset a 0 in these columns really means "not measured"
    for c in ["plas", "pres", "skin", "insu", "mass"]:
        if c in X.columns:
            X[c] = X[c].replace(0, np.nan)
    return X, y


def load_heart_data():
    try:
        d = fetch_openml(name="heart-statlog", version=1, as_frame=True, parser="auto")
        X = d.data.apply(pd.to_numeric, errors="coerce")
        y = d.target.astype(str).str.lower().isin(["present", "1", "yes"]).astype(int)
    except Exception:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
        df = pd.read_csv(url, header=None, names=cols, na_values="?")
        X, y = df.drop(columns="num"), (df["num"] > 0).astype(int)
    return X, y


LOADERS = {
    "Breast Cancer": load_breast_cancer_data,
    "Diabetes": load_diabetes_data,
    "Heart Disease": load_heart_data,
}


def get_dataset(name):
    cache = os.path.join(DATA_DIR, name.replace(" ", "_").lower() + ".csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        X, y = LOADERS[name]()
        df = X.copy()
        df["target"] = y.values
        df.to_csv(cache, index=False)
    return df.drop(columns="target"), df["target"]


# ---------------------------------------------------------------
# 2. MODELS
# ---------------------------------------------------------------
def get_models():
    if HAS_XGB:
        boost = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9,
                              eval_metric="logloss", random_state=SEED)
    else:
        boost = GradientBoostingClassifier(random_state=SEED)
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "XGBoost": boost,
    }


def make_pipeline(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", model)])


# ---------------------------------------------------------------
# 3. TRAIN + EVALUATE ONE DATASET
# ---------------------------------------------------------------
def run_dataset(name):
    print(f"\n{'=' * 60}\nDATASET: {name}\n{'=' * 60}")
    X, y = get_dataset(name)
    print(f"Patients: {len(X)} | Features: {X.shape[1]} | Disease cases: {int(y.sum())} ({y.mean() * 100:.1f}%)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted, probs = [], {}, {}
    for mname, model in get_models().items():
        pipe = make_pipeline(model)
        cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        prob = pipe.predict_proba(X_te)[:, 1]
        rows.append({
            "Dataset": name, "Model": mname,
            "CV Accuracy": round(cv_acc, 4),
            "Test Accuracy": round(accuracy_score(y_te, pred), 4),
            "Precision": round(precision_score(y_te, pred), 4),
            "Recall": round(recall_score(y_te, pred), 4),
            "F1": round(f1_score(y_te, pred), 4),
            "ROC-AUC": round(roc_auc_score(y_te, prob), 4),
        })
        fitted[mname], probs[mname] = pipe, prob
        print(f"  {mname:<20} acc={rows[-1]['Test Accuracy']:.3f}  "
              f"recall={rows[-1]['Recall']:.3f}  auc={rows[-1]['ROC-AUC']:.3f}")

    res = pd.DataFrame(rows)
    tag = name.replace(" ", "_").lower()

    # confusion matrices (all 4 models)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (mname, pipe) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_te, pipe.predict(X_te))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Healthy", "Disease"], yticklabels=["Healthy", "Disease"])
        ax.set_title(mname); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle(f"Confusion Matrices - {name}")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{tag}_confusion_matrices.png", dpi=130); plt.close()

    # ROC curves
    plt.figure(figsize=(6, 5))
    for mname, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        plt.plot(fpr, tpr, label=f"{mname} (AUC={roc_auc_score(y_te, prob):.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {name}"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{tag}_roc_curves.png", dpi=130); plt.close()

    # Random Forest feature importance (top 10)
    rf = fitted["Random Forest"].named_steps["model"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh", color="teal")
    plt.title(f"Top 10 Important Features - {name}"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{tag}_feature_importance.png", dpi=130); plt.close()
    print("  Top features:", ", ".join(imp.index[::-1][:5]))

    # save best model (by ROC-AUC)
    best = res.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    joblib.dump({"pipeline": fitted[best], "features": list(X.columns), "model_name": best},
                f"{OUT_DIR}/{tag}_best_model.joblib")
    print(f"  Best model: {best} (saved)")

    # demo: risk score for 3 test patients
    demo = fitted[best].predict_proba(X_te.head(3))[:, 1]
    for i, p in enumerate(demo):
        print(f"  Sample patient {i + 1}: disease probability = {p * 100:.1f}%  (actual: {'Disease' if y_te.iloc[i] else 'Healthy'})")
    return res


# ---------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------
def main():
    all_results = []
    for name in LOADERS:
        try:
            all_results.append(run_dataset(name))
        except Exception as e:
            print(f"\n[skipped] {name}: {e}")
            print("          (check your internet connection and run again)")

    if not all_results:
        print("No dataset could be loaded.")
        return
    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(f"{OUT_DIR}/results_summary.csv", index=False)
    print(f"\n{'=' * 60}\nFINAL RESULTS (all datasets)\n{'=' * 60}")
    print(final.to_string(index=False))

    pivot = final.pivot(index="Model", columns="Dataset", values="Test Accuracy")
    plt.figure(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.6, vmax=1.0)
    plt.title("Test Accuracy: Model vs Dataset")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/accuracy_comparison.png", dpi=130); plt.close()
    print(f"\nAll charts, results_summary.csv and saved models are in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
