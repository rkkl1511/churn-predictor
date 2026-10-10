"""Train the churn model and save the pipeline + metadata for the app."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_clean.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

TARGET = "Churn_Value"
NUM_COLS = ["Tenure_Months", "Monthly_Charges", "Total_Charges"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Total_Charges"] = pd.to_numeric(df["Total_Charges"], errors="coerce")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_pipeline(cat_cols, scale_pos_weight):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def main():
    X, y = load_data()
    cat_cols = [c for c in X.columns if c not in NUM_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipe = build_pipeline(cat_cols, scale_pos_weight)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
    }
    print("Test metrics:", metrics)

    # Metadata so the app builds its input form from the data, not hardcoded values
    metadata = {
        "feature_order": X.columns.tolist(),
        "num_cols": NUM_COLS,
        "cat_cols": cat_cols,
        "cat_options": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
        "num_ranges": {
            c: [float(X[c].min()), float(X[c].max()), float(X[c].median())]
            for c in NUM_COLS
        },
        "metrics": metrics,
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()