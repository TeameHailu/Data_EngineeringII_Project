import ray
from ray import tune
from sklearn.datasets import fetch_covtype
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_covtype(return_X_y=True)
X,y = resample(X, y, n_samples=10000, random_state=42)
# Split the dataset (optional but cleaner)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the trainable function for Ray Tune
def train_rf(config):
    model = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        n_jobs=1
    )
    score = cross_val_score(model, X_train, y_train, cv=2).mean()
    tune.report(mean_accuracy=score)

# Initialize Ray
ray.init(address="auto", ignore_reinit_error=True)

# Define hyperparameter search space
search_space = {
    "max_depth": tune.grid_search([10, 15]),
    "n_estimators": tune.grid_search([50, 100]),
    "ccp_alpha": tune.grid_search([0.0, 0.01])
}

# Run Ray Tune with grid search
tuner = tune.Tuner(
    train_rf,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="mean_accuracy",
        mode="max",
max_concurrent_trails=2
    )
)

results = tuner.fit()

# Get best result
best_result = results.get_best_result(metric="mean_accuracy", mode="max")
print("Best hyperparameters found were: ", best_result.config)
print("Best CV accuracy: ", best_result.metrics["mean_accuracy"])
