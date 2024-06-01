import os
import pickle
import click
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Update this URI as per your MLflow server
    mlflow.set_experiment("MLflow Quickstart with Hyperopt")

    # Define the space of hyperparameters to search
    space = {
        'max_depth': hp.choice('max_depth', range(5, 21)),
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 250]),
        'min_samples_split': hp.choice('min_samples_split', range(2, 11))
    }

    def objective(params):
        with mlflow.start_run():
            # Load data
            X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
            X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

            # Train model
            rf = RandomForestRegressor(
                max_depth=int(params['max_depth']),
                n_estimators=int(params['n_estimators']),
                min_samples_split=int(params['min_samples_split']),
                random_state=0
            )
            rf.fit(X_train, y_train)

            # Predict
            y_pred = rf.predict(X_val)

            # Calculate RMSE
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            # Log parameters and RMSE
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

            # Log the model for each run
            mlflow.sklearn.log_model(
                sk_model=rf,
                artifact_path="models",
                registered_model_name="RF-Model-Quickstart"
            )

            return {'loss': rmse, 'status': STATUS_OK}

    # Run hyperparameter optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50
    )

    print("Best hyperparameters:", best)


if __name__ == '__main__':
    run_train()
