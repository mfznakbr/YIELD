import os
import sys
import warnings
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
from preprocessing import preprocess_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil argumen CLI
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_train.xlsx")
    test_path = file_path.replace("data_train.xlsx", "testing.xlsx")

    print(f"Reading train data from: {file_path}")
    print(f"Reading test data from: {test_path}")

    # Load data
    train_df = pd.read_excel(file_path)
    test_df = pd.read_excel(test_path)

    target = "Ton"

    # Preprocessing
    preprocessing_path = os.path.join(os.path.dirname(__file__), "preprocessing.joblib")
    columns_path = os.path.join(os.path.dirname(__file__), "columns.csv")

    X_train, X_test, y_train, y_test = preprocess_data(
        train_df, test_df, target_column=target,
        save_path=preprocessing_path,
        file_path=columns_path
    )

    # Set tracking URI
    if "GITHUB_ACTIONS" in os.environ:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
    else:
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

    with mlflow.start_run():
        # Hyperparameter grid
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

        # Predict & evaluate
        y_pred = best_model.predict(X_test)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })

        mlflow.log_metrics({
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred),
            "test_score": best_model.score(X_test, y_test)
        })

        # Log preprocessing artifacts
        mlflow.log_artifact(preprocessing_path)
        mlflow.log_artifact(columns_path)

        # Log model dengan input_example dan signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_train, best_model.predict(X_train[:1]))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

    print("Modeling selesai dan berhasil dilog ke MLflow.")
