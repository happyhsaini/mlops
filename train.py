from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 1. Tracking URI aur Experiment set karein (port 6001 aur sqlite backend ke mutabiq)
mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("Iris-MLOps-Experiment")

# 2. Dataset load karein
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

n_estimators = 1
max_depth = 2
random_state = 0
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "random_state": random_state,
    "test_size": 0.25
}

# 3. Start a tracked training run using MLflow
with mlflow.start_run() as run:
    # Parameters log karein
    mlflow.log_params(model_parameters)

    # Model initialize aur train karein
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)  # Fix: X_trainy_train -> X_train, y_train

    # Predictions aur metrics calculate karein
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    MIN_TEST_ACCURACY = 0.90  # Minimum accuracy threshold for logging the model
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(f"Model rejected: test accuracy: {accuracy:.2f} is below {MIN_TEST_ACCURACY:.2f}. Model will not be logged.")

    print("Model Quality Gate Passed. Model is eligible for registration.")

    # Metrics log karein
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # MLflow ke andar model log/artifact save karein
    mlflow.sklearn.log_model(
        model,
        "iris-random-forest-model",
        registered_model_name="iris-classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Assign the champion alias to the newly registered model.
    client = MlflowClient()
    current_run_id = mlflow.active_run().info.run_id

    # Retrieve all versions registered under this model name.
    registered_versions = client.search_model_versions(
        "name='iris-classifier'"
    )
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )

    if current_model_version is None:
        raise RuntimeError(
            "Model was logged but its registered version could not be found."
        )

    client.set_registered_model_alias(
        name="iris-classifier",
        alias="champion",
        version=current_model_version.version
    )
    print(
        f"Champion alias assigned to model version "
        f"{current_model_version.version}."
    )

    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Accuracy logged to MLflow: {accuracy}")

# 4. Local Joblib file save (jaise aapke purane pipeline mein thi)
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_dir / "iris_model.joblib")
print(f"Model saved locally to {model_dir / 'iris_model.joblib'}")