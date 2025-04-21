from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif

class CreditCardFraudTrainingFlow(FlowSpec):
    n_estimators = Parameter('trees', default=100, type=int, help='Number of trees in the forest')
    max_depth = Parameter('depth', default=10, type=int, help='Maximum depth of the trees')
    random_seed = Parameter('seed', default=42, type=int, help='Random seed for reproducibility')
    top_n_features = Parameter('features', default=15, type=int, help='Number of top features to select')
    
    @step
    def start(self):
        """Starting point: Load and prepare data"""
        print("Loading credit card fraud dataset...")
        
        self.df = pd.read_csv("../data/creditcard.csv")
        
        # split features and target
        X = self.df.drop(['Class'], axis=1)
        y = self.df['Class']
        
        # norm Amount and Time features
        scaler = StandardScaler()
        X['Amount'] = scaler.fit_transform(X[['Amount']])
        X['Time'] = scaler.fit_transform(X[['Time']])
        
        # split data into train/val/test sets
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_val, self.y_train_val, test_size=0.25, 
            random_state=self.random_seed, stratify=self.y_train_val
        )
        
        print(f"Data loaded successfully. Training set size: {len(self.X_train)}")
        self.next(self.feature_selection)
    
    @step
    def feature_selection(self):
        """Select top features based on ANOVA F-value"""
        print(f"Selecting top {self.top_n_features} features...")
        
        # feature selection using SelectKBest
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
        
        # store feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': self.X_train_selected.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.next(self.evaluate_model)
    
    @step
    def evaluate_model(self):
        """Evaluate model performance"""
        print("Evaluating model performance...")
        
        # make predictions on validation set
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
    
    @step
    def register_model(self):
        """Register model with MLFlow"""
        print("Registering model with MLFlow...")
        
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('credit-card-fraud-detection')
        
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
            
            feature_importance_path = "../data/feature_importance.csv"
            self.feature_importance.to_csv(feature_importance_path, index=False)
            mlflow.log_artifact(feature_importance_path)
            
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="fraud_detection_model",
                registered_model_name="credit_card_fraud_detection"
            )
        
        self.next(self.end)
    
    @step
    def end(self):
        """Finish line"""
        print("Training flow completed successfully!")
        print(f"Model performance: F1={self.f1:.4f}")
        print(f"Selected features: {self.selected_features}")

if __name__ == '__main__':
    CreditCardFraudTrainingFlow()

# python src/trainingflow.py run --trees 100 --depth 10 --seed 42 --features 15