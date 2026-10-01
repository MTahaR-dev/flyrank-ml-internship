from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Locate the repo from this script's location.
REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "work" / "outputs"

# Load the five input features.
X = pd.read_csv(OUTPUTS / "capstone_features.csv")

# Load the separate answer column.
y = pd.read_csv(
    OUTPUTS / "capstone_target.csv"
)["future_decline_proxy"]

# Load identifiers used for grouping pages by client.
context = pd.read_csv(OUTPUTS / "capstone_page_context.csv")

# Load our saved training/testing client split.
split = json.loads(
    (OUTPUTS / "capstone_split.json").read_text(encoding="utf-8")
)

# Check that the files have matching row counts and expected features.
assert len(X) == len(y) == len(context)
assert list(X.columns) == split["feature_names"]

print("Feature table shape:", X.shape)
print("Target count:", len(y))
print("\nFirst five feature rows:")
print(X.head().to_string(index=False))

# Check that no client belongs to both groups.
train_clients = set(split["train_clients"])
test_clients = set(split["test_clients"])

assert train_clients.isdisjoint(test_clients)

# Identify each client's pages.
train_mask = context["client_hash_id"].isin(train_clients)
test_mask = context["client_hash_id"].isin(test_clients)

# Every page must belong to exactly one group.
assert (train_mask | test_mask).all()
assert not (train_mask & test_mask).any()

# Separate the inputs and answers.
X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()

X_test = X.loc[test_mask].copy()
y_test = y.loc[test_mask].copy()

print("\nTraining pages:", len(X_train))
print("Testing pages:", len(X_test))
print("Overlapping clients:", len(train_clients & test_clients))

# Connect input preparation and the model.
model = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )),
])

# Learn using training pages and their answers only.
model.fit(X_train, y_train)

print("\nModel training complete.")
print("Model:", type(model.named_steps["classifier"]).__name__)
print("Number of input features:", model.n_features_in_)

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# Keep client IDs outside the model inputs.
training_context = context.loc[
    train_mask, ["client_hash_id", "content_hash_id"]
].copy()

# Each prediction comes from a model that did not train on that client.
# cross_val_predict creates fresh copies of our entire pipeline.
cv_scores = cross_val_predict(
    model,
    X_train,
    y_train,
    groups=training_context["client_hash_id"],
    cv=GroupKFold(n_splits=4),
    method="predict_proba",
    n_jobs=1,
)[:, 1]

# Combine validation scores with identifiers and observed answers.
validation = training_context.copy()
validation["risk_score"] = cv_scores
validation["actual_decline"] = y_train

# Precision@20 requires at least 20 pages per client.
client_sizes = validation.groupby("client_hash_id")[
    "content_hash_id"
].transform("size")

eligible = validation.loc[client_sizes >= 20].copy()

# Rank within each client; resolve equal scores using content ID.
ranked = eligible.sort_values(
    ["client_hash_id", "risk_score", "content_hash_id"],
    ascending=[True, False, True],
)

top20 = ranked.groupby("client_hash_id").head(20)

# Give each eligible client equal weight.
precision20 = top20.groupby("client_hash_id")[
    "actual_decline"
].mean().mean()

random_precision20 = eligible.groupby("client_hash_id")[
    "actual_decline"
].mean().mean()

print("\nTRAINING-CLIENT CROSS-VALIDATION")
print("Eligible clients:", eligible["client_hash_id"].nunique())
print(f"Model mean Precision@20: {precision20:.1%}")
print(f"Expected random Precision@20: {random_precision20:.1%}")
print(f"ROC-AUC: {roc_auc_score(y_train, cv_scores):.3f}")

import numpy as np

baseline = json.loads(
    (OUTPUTS / "w04_baseline_metrics.json").read_text(encoding="utf-8")
)

test_context = context.loc[
    test_mask, ["client_hash_id", "content_hash_id"]
].copy()

# Confirm that we are using the baseline's test clients and page count.
assert sorted(test_context["client_hash_id"].unique()) == baseline["test_clients"]
assert len(X_test) == baseline["test_pages"]

# Our original model remains fitted on all training pages.
# Cross-validation used separate copies of it.
model_scores = model.predict_proba(X_test)[:, 1]

# Recover original integer impression counts from the log feature.
past_impressions = np.rint(np.expm1(X_test["log_impressions"]))
past_change = X_test["prior_week_change_pct"]

# Reproduce the frozen Week 4 rule.
rule_scores = np.where(
    (past_impressions >= baseline["rule_min_impressions"])
    & (past_change <= baseline["rule_max_prior_change_pct"]),
    np.log1p(past_impressions) * (-past_change / 100),
    0.0,
)

test_results = test_context.copy()
test_results["actual_decline"] = y_test
test_results["model_score"] = model_scores
test_results["rule_score"] = rule_scores

sizes = test_results.groupby("client_hash_id")[
    "content_hash_id"
].transform("size")

eligible_test = test_results.loc[sizes >= 20].copy()


def evaluate_ranking(score_column):
    ranked = eligible_test.sort_values(
        ["client_hash_id", score_column, "content_hash_id"],
        ascending=[True, False, True],
    )
    top20 = ranked.groupby("client_hash_id").head(20)

    return top20.groupby("client_hash_id")[
        "actual_decline"
    ].mean().mean()


comparison = pd.DataFrame([
    {
        "method": "Logistic regression",
        "precision_at_20_pct": 100 * evaluate_ranking("model_score"),
        "roc_auc": roc_auc_score(y_test, model_scores),
    },
    {
        "method": "Week 4 rule",
        "precision_at_20_pct": 100 * evaluate_ranking("rule_score"),
        "roc_auc": roc_auc_score(y_test, rule_scores),
    },
])

# Verify that the reproduced baseline matches its saved result.
assert np.isclose(
    evaluate_ranking("rule_score"),
    baseline["mean_client_precision_at_20"],
)

random_test_precision = eligible_test.groupby("client_hash_id")[
    "actual_decline"
].mean().mean()

print("\nTEST COMPARISON")
print(comparison.round(3).to_string(index=False))
print(f"Expected random Precision@20: {random_test_precision:.1%}")
print("Eligible test clients:", eligible_test["client_hash_id"].nunique())

# Model weights apply to standardized features.
weights = pd.DataFrame({
    "feature": X.columns,
    "weight": model.named_steps["classifier"].coef_[0],
})
weights["absolute_weight"] = weights["weight"].abs()

print("\nMODEL WEIGHTS")
print(
    weights.sort_values("absolute_weight", ascending=False)
    .round(4)
    .to_string(index=False)
)

# Find false positives within the model's top-20 recommendations.
model_top20 = (
    eligible_test.sort_values(
        ["client_hash_id", "model_score", "content_hash_id"],
        ascending=[True, False, True],
    )
    .groupby("client_hash_id")
    .head(20)
)

wrong_picks = (
    model_top20.loc[model_top20["actual_decline"].eq(0)]
    .nlargest(3, "model_score")
)

error_examples = X_test.loc[wrong_picks.index].copy()
error_examples["risk_score"] = wrong_picks["model_score"]
error_examples["actual_decline"] = wrong_picks["actual_decline"]

print("\nTHREE HIGH-SCORING FALSE POSITIVES")
print(error_examples.round(4).to_string(index=False))

import sys
import sklearn

# Save the trained pipeline locally.
# data/ is ignored by Git, so the model file stays out of the repo.
model_folder = REPO / "data" / "models"
model_folder.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "pipeline": model,
        "feature_names": list(X.columns),
        "split_seed": split["seed"],
    },
    model_folder / "capstone_logistic_model.joblib",
)

# Save aggregate results that we can commit.
metrics = {
    "model": "LogisticRegression",
    "model_parameters": model.named_steps["classifier"].get_params(),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "feature_names": list(X.columns),
    "training_pages": len(X_train),
    "test_pages": len(X_test),
    "primary_metric": "mean client Precision@20",
    "training_cross_validation": {
        "method": "4-fold GroupKFold by client",
        "precision_at_20": float(precision20),
        "expected_random_precision_at_20": float(random_precision20),
        "roc_auc": float(roc_auc_score(y_train, cv_scores)),
    },
    "test_comparison": comparison.to_dict(orient="records"),
    "test_expected_random_precision_at_20": float(random_test_precision),
    "eligible_test_clients": int(
        eligible_test["client_hash_id"].nunique()
    ),
    "standardized_feature_weights": {
        feature: float(weight)
        for feature, weight in zip(
            X.columns,
            model.named_steps["classifier"].coef_[0],
        )
    },
    "limitations": [
        "One March decision window, not a future-month evaluation.",
        "Test outcomes were previously examined in weekly assignments.",
        "Complete later coverage is required for evaluation.",
        "Decline prediction does not establish refresh benefit.",
        "Probability calibration has not been evaluated.",
    ],
}

metrics_path = OUTPUTS / "capstone_model_metrics.json"
metrics_path.write_text(
    json.dumps(metrics, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("\nSaved the trained model locally.")
print("Saved aggregate results:", metrics_path.name)