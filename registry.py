import mlflow
from mlflow.tracking import MlflowClient
import os
import pickle
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def evaluate_model(run_id, model_path, X_test, y_test):
    model_uri = f"runs:/{run_id}/{model_path}"
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse

# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Load test data
X_test, y_test = load_pickle("./output/test.pkl")

# Search the previous experiment's runs and get top 5 models based on validation RMSE
experiment_id = client.get_experiment_by_name("random-forest-hyperopt").experiment_id
runs = client.search_runs(experiment_ids=experiment_id, order_by=["metrics.rmse ASC"], max_results=5)

best_rmse = float('inf')
best_run_id = None

# Evaluate each model on the test set
for run in runs:
    run_id = run.info.run_id
    test_rmse = evaluate_model(run_id, "model", X_test, y_test)

    # Determine if this model has the best RMSE so far
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_run_id = run_id

# Register the best model
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "Best RandomForest Model"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"Registered model {model_name} with RMSE: {best_rmse}")
