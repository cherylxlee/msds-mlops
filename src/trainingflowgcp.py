from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, resources, retry, timeout, catch
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from google.cloud import storage
import io
import tempfile

@conda_base(libraries={
    'numpy': '1.23.5', 
    'pandas': '1.5.3', 
    'scikit-learn': '1.2.2', 
    'databricks-cli': '0.17.7',
    'mlflow': '2.3.1'
}, python='3.9.16')
class CreditCardFraudTrainingFlowGCP(FlowSpec):
    n_estimators = Parameter('trees', default=100, type=int, help='Number of trees in the forest')
    max_depth = Parameter('depth', default=10, type=int, help='Maximum depth of the trees')
    random_seed = Parameter('seed', default=42, type=int, help='Random seed for reproducibility')
    top_n_features = Parameter('features', default=15, type=int, help='Number of top features to select')
    
    @catch(var='start_exception')
    @retry(times=3)
    @timeout(minutes=10)
    @step
    def start(self):
        """Starting point: Load and prepare data"""
        print("Loading credit card fraud dataset...")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket("metaflow-data-metaflow-training-lab")
        blob = bucket.blob("data/creditcard.csv")
        content = blob.download_as_string()
        
        self.df = pd.read_csv(io.BytesIO(content))
        
        # split features and target
        X = self.df.drop(['Class'], axis=1)
        y = self.df['Class']
        
        # norm Amount and Time features
        scaler = StandardScaler()
        X['Amount'] = scaler.fit_transform(X[['Amount']])
        X['Time'] = scaler.fit_transform(X[['Time']])
        
        # split into train/val/test sets
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_val, self.y_train_val, test_size=0.25, 
            random_state=self.random_seed, stratify=self.y_train_val
        )
        
        print(f"Data loaded successfully. Training set size: {len(self.X_train)}")
        self.next(self.feature_selection)
    

    @catch(var='feature_selection_exception')
    @retry(times=3)
    @timeout(minutes=15)
    @resources(cpu=1, memory=2000)
    @kubernetes
    @step
    def feature_selection(self):
        """Select top features based on ANOVA F-value"""
        print(f"Selecting top {self.top_n_features} features...")
        
        # SelectKBest features
        selector = SelectKBest(f_classif, k=self.top_n_features)
        selector.fit(self.X_train, self.y_train)
        
        # selected feature indices
        feature_indices = selector.get_support(indices=True)
        self.selected_features = self.X_train.columns[feature_indices].tolist()
        
        # feature selection
        self.X_train_selected = self.X_train[self.selected_features]
        self.X_val_selected = self.X_val[self.selected_features]
        self.X_test_selected = self.X_test[self.selected_features]
        
        print(f"Selected features: {self.selected_features}")
        self.next(self.train_model)
    

    @catch(var='train_model_exception')
    @retry(times=3)
    @timeout(minutes=30)
    @resources(cpu=2, memory=4000)
    @kubernetes
    @step
    def train_model(self):
        """Train Random Forest model with best hyperparameters"""
        print("Training Random Forest model...")
        
        param_grid = {
            'n_estimators': [50, self.n_estimators],
            'max_depth': [5, self.max_depth]
        }
        
        # rf grid search
        rf = RandomForestClassifier(random_state=self.random_seed)
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(self.X_train_selected, self.y_train)
        
        self.best_params = grid_search.best_params_
        print(f"Best parameters: {self.best_params}")
        
        # train final model with best parameters
        self.model = RandomForestClassifier(**self.best_params, random_state=self.random_seed)
        self.model.fit(self.X_train_selected, self.y_train)
        
        # feature importance store
        self.feature_importance = pd.DataFrame({
            'Feature': self.X_train_selected.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.next(self.evaluate_model)


    @catch(var='evaluate_model_exception')
    @retry(times=3)
    @timeout(minutes=10)
    @resources(cpu=1, memory=2000)
    @kubernetes
    @step
    def evaluate_model(self):
        """Evaluate model performance"""
        print("Evaluating model performance...")
        
        # make predictions on val set
        y_pred_val = self.model.predict(self.X_val_selected)
        
        # metrics
        self.accuracy = accuracy_score(self.y_val, y_pred_val)
        self.precision = precision_score(self.y_val, y_pred_val)
        self.recall = recall_score(self.y_val, y_pred_val)
        self.f1 = f1_score(self.y_val, y_pred_val)
        
        print("Validation metrics:")
        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1 Score: {self.f1:.4f}")
        
        self.next(self.register_model)


    @catch(var='register_model_exception')
    @retry(times=3)
    @timeout(minutes=15)
    @step
    def register_model(self):
        """Register model with MLFlow"""
        print("Registering model with MLFlow...")
        
        mlflow.set_tracking_uri('https://mlflow-server-385494363170.us-west2.run.app')
        mlflow.set_experiment('credit-card-fraud-detection-gcp')
        
        with mlflow.start_run():
            mlflow.log_param("n_estimators", self.best_params['n_estimators'])
            mlflow.log_param("max_depth", self.best_params['max_depth'])
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("top_n_features", self.top_n_features)
            mlflow.log_param("selected_features", self.selected_features)
            
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.log_metric("precision", self.precision)
            mlflow.log_metric("recall", self.recall)
            mlflow.log_metric("f1", self.f1)
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                feature_importance_path = temp_file.name
                self.feature_importance.to_csv(feature_importance_path, index=False)
                mlflow.log_artifact(feature_importance_path)
            
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="fraud_detection_model",
                registered_model_name="credit_card_fraud_detection_gcp"
            )
        
        self.next(self.end)
    
    @step
    def end(self):
        """Finish line"""
        print("Training flow completed successfully!")
        print(f"Model performance: F1={self.f1:.4f}")
        print(f"Selected features: {self.selected_features}")
        
        # catch any exceptions
        if hasattr(self, 'start_exception') and self.start_exception:
            print(f"Error in start step: {self.start_exception}")
        if hasattr(self, 'feature_selection_exception') and self.feature_selection_exception:
            print(f"Error in feature selection step: {self.feature_selection_exception}")
        if hasattr(self, 'train_model_exception') and self.train_model_exception:
            print(f"Error in model training step: {self.train_model_exception}")
        if hasattr(self, 'evaluate_model_exception') and self.evaluate_model_exception:
            print(f"Error in model evaluation step: {self.evaluate_model_exception}")
        if hasattr(self, 'register_model_exception') and self.register_model_exception:
            print(f"Error in model registration step: {self.register_model_exception}")

if __name__ == '__main__':
    CreditCardFraudTrainingFlowGCP()
