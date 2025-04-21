from metaflow import FlowSpec, step, Parameter, Flow, JSONType
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CreditCardFraudScoringFlow(FlowSpec):
    transaction_data = Parameter('data', type=JSONType, required=True, 
                               help='Transaction data for fraud detection (JSON array)')
    model_version = Parameter('version', default='latest', 
                            help='Version of the model to use')
    
    @step
    def start(self):
        """Starting point: Load model and data"""
        print(f"Input transaction: {self.transaction_data}")
        
        # valid. input format
        if not isinstance(self.transaction_data, dict) and not isinstance(self.transaction_data, list):
            raise ValueError("Transaction data must be a dictionary or list of features")
            
        # get the latest training flow run to access metadata
        self.train_flow = Flow('CreditCardFraudTrainingFlow').latest_run
        print(f"Using model from run: {self.train_flow.pathspec}")
        
        # get selected features from training flow
        self.selected_features = self.train_flow['feature_selection'].task.data.selected_features
        print(f"Using features: {self.selected_features}")
        
        # load the model from MLFlow
        print(f"Loading model version: {self.model_version}")
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        self.model = mlflow.sklearn.load_model(f"models:/credit_card_fraud_detection/{self.model_version}")
        
        # prepare input data
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """Preprocess the input transaction data"""
        print("Preprocessing transaction data...")
        
        # convert input to DataFrame
        if isinstance(self.transaction_data, list):
            # multiple transactions
            self.transaction_df = pd.DataFrame([self.transaction_data])
        else:
            # single transaction as dictionary
            self.transaction_df = pd.DataFrame([self.transaction_data])
        
        # ensure all necessary columns exist
        required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        for col in required_columns:
            if col not in self.transaction_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # normalize Amount and Time
        scaler = StandardScaler()
        self.transaction_df['Amount'] = scaler.fit_transform(self.transaction_df[['Amount']])
        self.transaction_df['Time'] = scaler.fit_transform(self.transaction_df[['Time']])
        
        # select features used during training
        self.processed_data = self.transaction_df[self.selected_features]
        
        self.next(self.predict)
    
    @step
    def predict(self):
        """Make prediction using the loaded model"""
        print("Making prediction...")
        
        # predict fraud probability
        self.probabilities = self.model.predict_proba(self.processed_data)
        self.prediction = self.model.predict(self.processed_data)[0]
        
        # get fraud probability from prediction
        fraud_probability = self.probabilities[0][1] if self.prediction == 1 else 1 - self.probabilities[0][1]
        self.fraud_probability = float(fraud_probability)
        
        self.next(self.end)
    
    @step
    def end(self):
        """Finish line"""
        print("Prediction complete!")
        print(f"Fraud detected: {'Yes' if self.prediction == 1 else 'No'}")
        print(f"Fraud probability: {self.fraud_probability:.4f}")
        
        self.result = {
            "is_fraud": bool(self.prediction),
            "fraud_probability": self.fraud_probability
        }

if __name__ == '__main__':
    CreditCardFraudScoringFlow()

# python scoringflow.py run --data '{"Time": 10000, "Amount": 5.0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47, "V17": -0.11, "V18": -0.15, "V19": 0.06, "V20": 0.17, "V21": 0.15, "V22": 0.04, "V23": 0.22, "V24": 0.17, "V25": 0.21, "V26": 0.12, "V27": 0.02, "V28": 0.01}'
