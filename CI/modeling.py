import os
from pathlib import Path
import mlflow
import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import preprocess_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil argumen CLI
    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_train.xlxs")
    data = pd.read_excel(file_path)
    # Load data
    train_df = pd.read_excel(data)
    test_df = pd.read_excel("testing.xlsx")
    target = "Ton"

    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(
        train_df, test_df, target_column=target,
        save_path="preprocessing.joblib",
        file_path="columns.csv"
    )

    # MLflow run
    with mlflow.start_run():
        # Setup dan training model
        param_grid = {
            "n_estimators": [n_estimators],
            "max_depth": [max_depth],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.7],
        }

        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluasi
        y_pred = best_model.predict(X_test)
        mlflow.log_metrics({
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred)
        })
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })

        # Simpan model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train[:1]
        )
        accuracy = best_model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)


