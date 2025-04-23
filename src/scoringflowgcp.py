from metaflow import FlowSpec, step, Parameter, Flow, JSONType, conda_base, kubernetes, resources, retry, timeout, catch
import mlflow
import pandas as pd

@conda_base(libraries={
    'numpy': '1.23.5', 
    'pandas': '1.5.3', 
    'scikit-learn': '1.2.2', 
    'databricks-cli': '0.17.7',
    'mlflow': '2.3.1'
}, python='3.9.16')
class CreditCardFraudScoringFlowGCP(FlowSpec):
    transaction_data = Parameter('data', type=JSONType, required=True, 
                               help='Transaction data for fraud detection (JSON array)')
    model_version = Parameter('version', default='latest', 
                            help='Version of the model to use')
    
    @catch(var='start_exception')
    @retry(times=3)
    @timeout(minutes=5)
    @step
    def start(self):
        """Starting point: Load model and data"""
        print(f"Input transaction: {self.transaction_data}")
        
        # validate input format
        if not isinstance(self.transaction_data, dict) and not isinstance(self.transaction_data, list):
            raise ValueError("Transaction data must be a dictionary or list of features")
            
        # get only successful training flow run to access metadata
        for run in Flow('CreditCardFraudTrainingFlowGCP'):
            if run.successful:
                self.train_flow = run
                print(f"Using successful model from run: {self.train_flow.pathspec}")
                break
        else:
            raise ValueError("No successful training flow runs found")
        print(f"Using model from run: {self.train_flow.pathspec}")
        
        # get selected features from training flow
        self.selected_features = self.train_flow['feature_selection'].task.data.selected_features
        print(f"Using features: {self.selected_features}")
        
        # load the model from MLFlow
        print(f"Loading model version: {self.model_version}")
        mlflow.set_tracking_uri('https://mlflow-server-385494363170.us-west2.run.app')
        self.model = mlflow.sklearn.load_model(f"models:/credit_card_fraud_detection_gcp/{self.model_version}")
        
        # prepare input data
        self.next(self.preprocess_data)


    @catch(var='preprocess_exception')
    @retry(times=3)
    @timeout(minutes=5)
    @resources(cpu=1, memory=2000)
    @kubernetes
    @step
    def preprocess_data(self):
        """Preprocess the input transaction data"""
        print("Preprocessing transaction data...")
        
        # convert input to DataFrame
        if isinstance(self.transaction_data, list):
            # multiple transactions
            self.transaction_df = pd.DataFrame(self.transaction_data)
        else:
            # single transaction as dictionary
            self.transaction_df = pd.DataFrame([self.transaction_data])
        
        # ensure all necessary columns exist
        required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        for col in required_columns:
            if col not in self.transaction_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # normalize Amount and Time
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.transaction_df['Amount'] = scaler.fit_transform(self.transaction_df[['Amount']])
        self.transaction_df['Time'] = scaler.fit_transform(self.transaction_df[['Time']])
        
        # select features used during training
        self.processed_data = self.transaction_df[self.selected_features]
        
        self.next(self.predict)


    @catch(var='predict_exception')
    @retry(times=3)
    @timeout(minutes=5)
    @resources(cpu=1, memory=2000)
    @kubernetes
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
        
        # catch any exceptions
        if hasattr(self, 'start_exception') and self.start_exception:
            print(f"Error in start step: {self.start_exception}")
        if hasattr(self, 'preprocess_exception') and self.preprocess_exception:
            print(f"Error in preprocess step: {self.preprocess_exception}")
        if hasattr(self, 'predict_exception') and self.predict_exception:
            print(f"Error in predict step: {self.predict_exception}")
            
        self.result = {
            "is_fraud": bool(self.prediction),
            "fraud_probability": self.fraud_probability
        }

if __name__ == '__main__':
    CreditCardFraudScoringFlowGCP()
