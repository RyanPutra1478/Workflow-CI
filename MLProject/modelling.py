import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Wine_Quality_Prediction")
    mlflow.autolog()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'dataset_preprocessing', 'X_train.csv')
    
    if not os.path.exists(train_path):
        print(f"Data tidak ditemukan di {train_path}")
        exit(1)
        
    X_train = pd.read_csv(train_path)
    y_train = pd.read_csv(os.path.join(base_dir, 'dataset_preprocessing', 'y_train.csv')).values.ravel()
    
    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        rf.fit(X_train, y_train)
        print("Training selesai dengan mlflow.autolog()!")
