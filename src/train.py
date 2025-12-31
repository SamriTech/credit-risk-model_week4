import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, 
    f1_score, accuracy_score
)

def save_learning_curve(model, X, y, model_name):
    """
    Generates a learning curve plot and saves it to the notebooks folder.
    This helps prove the model is generalizing and not overfitting.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, scoring='roc_auc', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Training Examples")
    plt.ylabel("ROC-AUC Score")
    plt.legend(loc="best")
    plt.grid()

    # Ensure the directory exists
    os.makedirs('notebooks', exist_ok=True)
    plot_path = f"notebooks/{model_name}_learning_curve.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

import os

def train_and_track_models():
    # 1. Setup Data
    df = pd.read_csv('data/processed/final_data.csv')
    
    # --- RIGOROUS DATA LEAKAGE FIX ---
    # We drop any column that 'cheats' or is a result of the risk (like FraudResult)
    leakage_cols = [
        'is_high_risk', 
        'FraudResult',      # Direct Leak
        'Amount', 'Value', 
        'Total_Amount', 'Average_Amount', 'Std_Amount',
        'TransactionId', 'CustomerId', 'Transaction_Count' 
    ]
    
    X = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    y = df['is_high_risk']
    
    # Selecting only numeric features
    X = X.select_dtypes(include=[np.number])
    
    print(f"Training on CLEAN behavioral features: {X.columns.tolist()}")
    
    # 2. Data Preparation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model Definitions
    model_configs = {
        "Logistic_Regression": (LogisticRegression(max_iter=1000, solver="liblinear"), {
            'C': [0.1, 1, 10]
        }),
        "Decision_Tree": (DecisionTreeClassifier(random_state=42), {
            'max_depth': [5, 10, 20]
        }),
        "Random_Forest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        }),
        "Gradient_Boosting": (GradientBoostingClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        })
    }

    best_roc_auc = 0
    best_run_id = None

    # Set the experiment
    mlflow.set_experiment("Credit_Risk_Evaluation_Final_Clean")

    for model_name, (model, params) in model_configs.items():
        with mlflow.start_run(run_name=model_name) as run:
            print(f"Tuning {model_name}...")
            
            grid = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            
            # Get probabilities for ROC-AUC
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                y_proba = best_model.decision_function(X_test)
                
            y_pred = best_model.predict(X_test)

            # 4. Evaluation Metrics
            metrics = {
                "ROC_AUC": roc_auc_score(y_test, y_proba),
                "Recall": recall_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "F1_Score": f1_score(y_test, y_pred),
                "Accuracy": accuracy_score(y_test, y_pred)
            }

            # 5. Logging to MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            
            # --- OVERFITTING CHECK: Log Learning Curve ---
            plot_file = save_learning_curve(best_model, X_train, y_train, model_name)
            mlflow.log_artifact(plot_file)
            
            mlflow.sklearn.log_model(best_model, "model")
            
            print(f"{model_name} Complete. ROC-AUC: {metrics['ROC_AUC']:.4f}")

            if metrics["ROC_AUC"] > best_roc_auc:
                best_roc_auc = metrics["ROC_AUC"]
                best_run_id = run.info.run_id

    # 6. Model Registry
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "High_Risk_Predictor_Model_Final")
        print(f"\nModel Registry: Best CLEAN model registered: {best_run_id}")

if __name__ == "__main__":
    train_and_track_models()