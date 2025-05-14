import ray
import numpy as np
import time
from ray import tune
from ray.air import session
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = fetch_covtype()
X, y = data.data, data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Put datasets in Ray object store
X_train_id = ray.put(X_train)
y_train_id = ray.put(y_train)

# Baseline model
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_score = accuracy_score(y_test, baseline_model.predict(X_test))
print(f"✅ Baseline accuracy: {baseline_score:.4f}")


# Ray Tune trainable function (reads from object store)
def rf_trainable(config):
    X_t = ray.get(X_train_id)
    y_t = ray.get(y_train_id)

    model = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        random_state=42,
        n_jobs=1
    )

    score = cross_val_score(model, X_t, y_t, cv=3).mean()
    session.report({"mean_accuracy": score})


# Search space
search_space = {
    "max_depth": tune.grid_search([10, 20, 30]),
    "n_estimators": tune.grid_search([50, 100]),
    "ccp_alpha": tune.grid_search([0.0, 0.01, 0.1]),
}

if __name__ == "__main__":
    ray.init(address="auto", ignore_reinit_error=True)

    start = time.time()
    tuner = tune.Tuner(
        rf_trainable,
        param_space=search_space,
        run_config=tune.RunConfig(
            name="rf_tuning",
            storage_path="file:///home/ubuntu/Data_EngineeringII_Project/Project_IV/ray_results",
            verbose=1
        ),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            reuse_actors=True
        )
    )

    results = tuner.fit()
    end = time.time()
    print(f"⏱️ Tuning time: {end - start:.2f} seconds")

    # Evaluate best model
    best_result = results.get_best_result(metric="mean_accuracy", mode="max")
    best_config = best_result.config
    print("🎯 Best hyperparameters:", best_config)

    final_model = RandomForestClassifier(
        max_depth=best_config["max_depth"],
        n_estimators=best_config["n_estimators"],
        ccp_alpha=best_config["ccp_alpha"],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    final_score = accuracy_score(y_test, final_model.predict(X_test))
    print(f"✅ Final accuracy after tuning: {final_score:.4f}")

