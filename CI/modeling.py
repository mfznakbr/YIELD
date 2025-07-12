import os
from pathlib import Path

# Fix path untuk GitHub Actions
mlruns_path = Path(__file__).parent / "mlruns"
mlruns_path.mkdir(exist_ok=True)

os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_path.absolute()}"
os.environ["MLFLOW_ARTIFACTS_URI"] = f"file://{mlruns_path.absolute()}"
print(f"Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")  # Debugging

# ... (kode lainnya tetap sama)
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

    # === Ambil argumen CLI ===
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    data_path = sys.argv[3] if len(sys.argv) > 3 else "data_train.xlsx"

    # === Load data ===
    train_df = pd.read_excel(data_path)
    test_df = pd.read_excel("testing.xlsx")
    target = "Ton"

    # === Preprocessing ===
    pipeline_path = "preprocessing.joblib"
    file_path = "columns.csv"
    X_train, X_test, y_train, y_test = preprocess_data(
        train_df, test_df, target_column=target,
        save_path=pipeline_path,
        file_path=file_path
    )

    # === Mulai MLflow run ===
    with mlflow.start_run():
        # Setup grid
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

        # Training
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluasi
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Logging manual
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # Simpan model
        input_example = X_train[:1]
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example
        )

        print("✅ Best Params:", grid_search.best_params_)
        print("📈 Test R2 Score:", round(r2, 3))
        print("🚀 MLflow run completed.")
