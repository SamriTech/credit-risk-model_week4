from fastapi import FastAPI, HTTPException
import os
import mlflow
import mlflow.sklearn
import pandas as pd

try:
    from src.api.pydantic_models import PredictionRequest, PredictionResponse
except ImportError:
    from pydantic_models import PredictionRequest, PredictionResponse


app = FastAPI(title="Credit Risk Prediction API")


# Try model registry first, then fall back to local artifact under /app/mlruns
MODEL_NAME = os.environ.get("MODEL_NAME", "High_Risk_Predictor_Model_Final")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////app/mlflow_data/mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None
try:
    # Try registry
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print("Loaded model from registry")
except Exception:
    # Fallback: load first local model artifact found
    import glob

    candidates = glob.glob("/app/mlruns/**/artifacts", recursive=True)
    for c in candidates:
        if os.path.isfile(os.path.join(c, "MLmodel")):
            try:
                print(f"Loading local model at: {c}")
                model = mlflow.sklearn.load_model(f"file://{c}")
                print("Loaded model from local artifacts")
                break
            except Exception as e:
                print(f"Failed to load local model at {c}: {e}")


@app.get("/")
def health_check():
    return {"status": "Online", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert payload to dataframe; exclude CustomerId if present
    data = pd.DataFrame([payload.dict()])
    if "CustomerId" in data.columns:
        data = data.drop(columns=["CustomerId"])

    try:
        proba = model.predict_proba(data)[0][1]
        pred = model.predict(data)[0]
        return {"risk_score": float(proba), "is_high_risk": bool(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))