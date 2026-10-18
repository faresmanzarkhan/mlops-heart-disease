import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def train_models():
    data_path = "data/processed/heart_cleaned.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    mlflow.set_experiment("heart-disease-classification")
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    os.makedirs("models", exist_ok=True)
    best_auc = 0
    best_model_name = None
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", model)
            ])
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_aucs, cv_accs, cv_precisions, cv_recalls = [], [], [], []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                pipeline.fit(X_fold_train, y_fold_train)
                preds = pipeline.predict(X_fold_val)
                probs = pipeline.predict_proba(X_fold_val)[:, 1]
                
                cv_aucs.append(roc_auc_score(y_fold_val, probs))
                cv_accs.append(accuracy_score(y_fold_val, preds))
                cv_precisions.append(precision_score(y_fold_val, preds, zero_division=0))
                cv_recalls.append(recall_score(y_fold_val, preds, zero_division=0))
            
            # Train final on full train split
            pipeline.fit(X_train, y_train)
            test_preds = pipeline.predict(X_test)
            test_probs = pipeline.predict_proba(X_test)[:, 1]
            
            test_acc = accuracy_score(y_test, test_preds)
            test_recall = recall_score(y_test, test_preds, zero_division=0)
            mean_cv_auc = np.mean(cv_aucs)
            
            # Log params & metrics
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            
            mlflow.log_metric("cv_roc_auc", mean_cv_auc)
            mlflow.log_metric("cv_precision", np.mean(cv_precisions))
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_recall", test_recall)
            
            mlflow.sklearn.log_model(pipeline, "model")
            
            if mean_cv_auc > best_auc:
                best_auc = mean_cv_auc
                best_model_name = name
                joblib.dump(pipeline, "models/best_model.joblib")
                
    print(f"Training complete. Best model: {best_model_name} with CV ROC-AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_models()
