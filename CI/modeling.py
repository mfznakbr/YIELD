import mlflow
import pandas as pd
import numpy as np
import os
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

    # === Baca data ===
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

    # === Setup MLflow ===
    # Jangan set URI agar default ke ./mlruns (portable untuk GitHub Actions)
    #mlflow.set_experiment("Prediksi Panen Tebu (GridSearch RF)")

    with mlflow.start_run(nested=True):
        # Auto-log semua artifact dan metric
        mlflow.autolog()

        # === Setup hyperparameter grid ===
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

        # === Training ===
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # === Evaluasi ===
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # === Logging manual ===
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # === Simpan model final ===
        input_example = X_train[0].reshape(1, -1)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="random_forest_final_model",
            input_example=input_example,
            registered_model_name="predict_sugarcane"
        )

        print("✅ Best Params:", grid_search.best_params_)
        print("📈 Test R2 Score:", round(r2, 3))
        print("🚀 MLflow run completed.")
