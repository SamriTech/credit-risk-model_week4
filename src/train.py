import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

import mlflow
import mlflow.sklearn


BASE_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = BASE_DIR / "mlflow_data" / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("Credit_Risk_Customer_Level")


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime
    if "TransactionStartTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    elif "TransactionTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionTime"])
    else:
        raise ValueError("No transaction time column found in data")

    # Handle empty dataframe gracefully: return empty feature frame with expected columns
    if df.empty:
        cols = [
            "CustomerId", "Transaction_Count", "Total_Amount",
            "Average_Amount", "Std_Amount", "RecencyDays"
        ]
        return pd.DataFrame(columns=cols)

    # Define snapshot date as one day after last transaction
    snapshot = df["TransactionStartTime"].max() + timedelta(days=1)

    # Group by customer and compute RFM-like features
    agg = df.groupby("CustomerId").agg(
        Transaction_Count=("Amount", "count"),
        Total_Amount=("Amount", "sum"),
        Average_Amount=("Amount", "mean"),
        Std_Amount=("Amount", "std"),
        First_Transaction=("TransactionStartTime", "min"),
        Last_Transaction=("TransactionStartTime", "max"),
    ).reset_index()

    agg["RecencyDays"] = (snapshot - agg["Last_Transaction"]).dt.days
    agg["ActiveDays"] = (agg["Last_Transaction"] - agg["First_Transaction"]).dt.days + 1
    # Keep Transaction_Count as the standard RFM 'Frequency' measure
    # Avoid creating a derived 'Frequency' that is highly correlated with Transaction_Count
    # (which can destabilize coefficients in linear models).

    # Replace NaN std with 0
    agg["Std_Amount"] = agg["Std_Amount"].fillna(0.0)

    # Select numeric features
    features = agg[[
        "CustomerId", "Transaction_Count", "Total_Amount",
        "Average_Amount", "Std_Amount", "RecencyDays"
    ]]

    return features


def train_customer_level_model():
    df = pd.read_csv(BASE_DIR / "data/processed/final_data.csv")

    # Build features and target at customer level
    features = build_customer_features(df)

    # Target: take max per customer (assumes is_high_risk labeled on transactions)
    if "is_high_risk" in df.columns:
        target = df.groupby("CustomerId")["is_high_risk"].max().reset_index()
        pos_count = int(target["is_high_risk"].sum())
    else:
        # No explicit target available — create empty target to be filled by fallback
        target = pd.DataFrame({"CustomerId": features["CustomerId"], "is_high_risk": 0})
        pos_count = 0

    # If there are too few positive examples, create a proxy target by RFM ranking
    # This helps training proceed when the original labeling yields <5% positives
    min_pos = max(2, int(0.05 * len(features)))
    if pos_count < min_pos:
        print(f"Insufficient positive examples ({pos_count}); creating proxy target using RFM percentile (min_pos={min_pos}).")

        f = features.copy()
        # Higher RecencyDays -> less engaged -> higher risk
        f["recency_rank"] = f["RecencyDays"].rank(pct=True)
        # Lower Total_Amount -> potentially riskier (use inverse rank)
        f["amount_rank"] = f["Total_Amount"].rank(pct=True)

        # Combined simple risk score: weight recency more than amount
        f["risk_score"] = f["recency_rank"] * 0.6 + (1 - f["amount_rank"]) * 0.4

        # Label top 10% risk as high risk (adjustable)
        cutoff_quantile = 0.90
        cutoff = f["risk_score"].quantile(cutoff_quantile)
        proxy = f[["CustomerId", "risk_score"]].copy()
        proxy["is_high_risk"] = (proxy["risk_score"] >= cutoff).astype(int)

        new_pos = int(proxy["is_high_risk"].sum())
        print(f"Proxy labeling created {new_pos} positives (cutoff quantile={cutoff_quantile}).")

        target = proxy[["CustomerId", "is_high_risk"]]

    data = features.merge(target, on="CustomerId", how="inner")

    X = data.drop(columns=["CustomerId", "is_high_risk"])  # customer-level predictors
    y = data["is_high_risk"].astype(int)

    # Split by customer
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    param_grid = {"clf__C": [0.1, 1.0, 10.0]}

    with mlflow.start_run(run_name="Logistic_Customer_RFM") as run:
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_

        # Predictions
        if hasattr(best, "predict_proba"):
            y_proba = best.predict_proba(X_test)[:, 1]
        else:
            y_proba = best.decision_function(X_test)

        y_pred = best.predict(X_test)

        metrics = {
            "ROC_AUC": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred)
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(best, "model")

        # Register model
        run_id = run.info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", "High_Risk_Predictor_Model_Final")

        print("Training complete. Metrics:", metrics)


if __name__ == "__main__":
    train_customer_level_model()