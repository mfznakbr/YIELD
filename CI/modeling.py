import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from preprocessing import preprocess_data

train_df = pd.read_excel("data_train.xlsx")
test_df = pd.read_excel("testing.xlsx")
target = 'Ton'

pipeline_path = "preprocessing.joblib"
file_path = "columns.csv"


X_train, X_test, y_train, y_test = preprocess_data(
    train_df, test_df, target_column='Ton', 
    save_path=pipeline_path,
    file_path=file_path
)


# ==== 2. Preprocess ====

# ==== 3. Setup MLflow ====
mlflow.set_tracking_uri("http://127.0.0.1:5005")
mlflow.set_experiment("Prediksi Panen Tebu (GridSearch RF)")

with mlflow.start_run():
    mlflow.autolog()

    # ==== 4. Hyperparameter Grid ====
    param_grid = {
        'n_estimators': [45, 50, 100],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.7]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    # ==== 5. Train ====
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # ==== 6. Predict & Evaluate ====
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_r2_score", r2)

    print("✅ Best Parameters:", best_params)
    print("📊 Best CV Score (neg MSE):", grid_search.best_score_)
    print(f"📈 Test RMSE: {rmse:.2f}")
    print(f"📈 Test R2 Score: {r2:.3f}")

    # ==== 7. Log Model ====
    input_example = X_train[0].reshape(1, -1)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="random_forest_final_model",
        input_example=input_example,
        registered_model_name="predict sugarcane"
    )

print("🚀 MLflow run completed.")
