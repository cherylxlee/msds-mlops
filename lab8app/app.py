from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import os

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="0.1",
)

class TransactionData(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 0,
                "Amount": 100.0,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215
            }
        }

# response schema
class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float

# global variables
model = None
selected_features = None
scaler = StandardScaler()

# load model at startup
@app.on_event("startup")
async def startup_event():
    global model, selected_features
    try:
        mlflow_db_path = os.path.abspath("../src/mlflow.db")
        mlflow_uri = f'sqlite:///{mlflow_db_path}'
        logger.info(f"Setting MLFlow tracking URI to: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
        
        model_path = "models:/credit_card_fraud_detection/latest"
        logger.info(f"Loading model from: {model_path}")
        model = mlflow.sklearn.load_model(model_path)
        logger.info("Model loaded successfully from MLFlow")
        
        # log the model's expected feature names
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model's feature_names_in_: {model.feature_names_in_}")
        
        client = mlflow.tracking.MlflowClient()
        
        # find latest run in the experiment
        experiment = client.get_experiment_by_name("credit-card-fraud-detection")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                # get selected features from run params
                run_id = runs[0].info.run_id
                selected_features_str = client.get_run(run_id).data.params.get("selected_features")
                if selected_features_str:
                    # convert string representation to actual list
                    selected_features = eval(selected_features_str)
                    logger.info(f"Selected features from MLflow: {selected_features}")
                else:
                    # if no selected features found, attempt to use model's feature_names_in_
                    if hasattr(model, 'feature_names_in_'):
                        selected_features = model.feature_names_in_.tolist()
                        logger.info(f"Using model's feature_names_in_: {selected_features}")
                    else:
                        # default features if not found
                        selected_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                        logger.info(f"Using default features: {selected_features}")
            else:
                logger.warning("No runs found in MLflow")
                # try to use model's feature_names_in_
                if hasattr(model, 'feature_names_in_'):
                    selected_features = model.feature_names_in_.tolist()
                    logger.info(f"Using model's feature_names_in_: {selected_features}")
                else:
                    selected_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                    logger.info(f"No runs found, using default features: {selected_features}")
        else:
            logger.warning("Experiment not found in MLflow")
            # try to use model's feature_names_in_
            if hasattr(model, 'feature_names_in_'):
                selected_features = model.feature_names_in_.tolist()
                logger.info(f"Using model's feature_names_in_: {selected_features}")
            else:
                selected_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                logger.info(f"Experiment not found, using default features: {selected_features}")

    except Exception as e:
        logger.error(f"Error loading model from MLFlow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# root endpoint
@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API"}

def preprocess_data(df):
    logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
    logger.info(f"Selected features: {selected_features}")
    
    # normalize Amount and Time 
    if 'Amount' in df.columns and 'Amount' in selected_features:
        df = df.copy()
        df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    if 'Time' in df.columns and 'Time' in selected_features:
        if 'Amount' not in selected_features:
            df = df.copy()
        df['Time'] = scaler.fit_transform(df[['Time']])
    
    # ensure all selected features exist in the DataFrame
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    result = df[selected_features].copy()
    logger.info(f"Preprocessed DataFrame columns: {result.columns.tolist()}")
    return result

# prediction endpoint for a single transaction
@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionData):
    try:
        df = pd.DataFrame([data.dict()])
        logger.info(f"Received prediction request with data shape: {df.shape}")
        
        processed_data = preprocess_data(df)
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        
        probabilities = model.predict_proba(processed_data)
        prediction = model.predict(processed_data)[0]
        
        fraud_probability = probabilities[0][1] if prediction == 1 else 1 - probabilities[0][1]
        
        # log prediction
        logger.info(f"Prediction: is_fraud={bool(prediction)}, probability={fraud_probability}")
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(fraud_probability)
        }
    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
