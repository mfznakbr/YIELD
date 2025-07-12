import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import sys
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import preprocess_data

# Tracking URI aman untuk CI/CD GitHub Actions
if "GITHUB_ACTIONS" in os.environ:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "data_train.xlsx")

    print(f"Reading train data from: {file_path}")

    # Load data
    train_df = pd.read_excel(file_path)
    test_df = pd.read_excel(os.path.join(os.path.dirname(__file__), "testing.xlsx"))
    target = "Ton"

    X_train, X_test, y_train, y_test = preprocess_data(
        train_df, test_df, target_column=target,
        save_path=os.path.join(os.path.dirname(__file__), "preprocessing.joblib"),
        file_path=os.path.join(os.path.dirname(__file__), "columns.csv")
    )

    # -- BLOK INI SUDAH DIPERBAIKI INDENTASINYA --
    param_grid = {
        "n_estimators": [n_estimators],
        "max_depth": [max_depth],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7],
    }

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
    mlflow.log_metrics({
        "test_mse": mean_squared_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_r2": r2_score(y_test, y_pred),
        "accuracy": best_model.score(X_test, y_test)
    })

    mlflow.sklearn.log_model(
        sk_model=best_model,
        input_example=X_train[:1],
        name="model"
    )
