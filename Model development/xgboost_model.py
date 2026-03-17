import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_models import compute_rolling_features as compute_rolling_features_linear
from train_models import compute_prior_season_features

# =============================================================================
# Prepare 2022-2025 training data
# =============================================================================

df_all = pd.read_csv("../Historical/scores_all_years_adjusted.csv")
df = df_all[df_all["Event"] != "AA"].copy()
df = df[df["Week"] <= 12]
df = df.sort_values(by=["GymnastID", "Event", "Year", "Date"])

def compute_rolling_features(group):
    group["High_Score"] = group["score_adj"].expanding().max().shift(1)
    group["Average"]    = group["score_adj"].expanding().mean().shift(1)
    group["score_1"]    = group["score_adj"].shift(1)
    group["score_2"]    = group["score_adj"].shift(2)
    return group

df["GymnastSeason"] = df["GymnastID"].astype(str) + "_" + df["Year"].astype(str)
year_index = df["Year"]
event_index = df["Event"]
df = df.groupby(["GymnastSeason", "Event"], group_keys=False).apply(compute_rolling_features, include_groups=False)
df["Year"] = year_index
df["Event"] = event_index

df = df.dropna(subset=["score_2"])
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Add prior season features (career_high, prev_season_avg, has_prior_season_data)
prior_season_frames = []
for yr in sorted(df["Year"].unique()):
    psf = compute_prior_season_features(df_all, yr)
    psf["Year"] = yr
    prior_season_frames.append(psf)
prior_season_df = pd.concat(prior_season_frames, ignore_index=True)
df = df.merge(prior_season_df, on=["GymnastID", "Event", "Year"], how="left")
for col in ["career_high", "prev_season_avg"]:
    event_means = df.groupby("Event")[col].transform("mean")
    df[col] = df[col].fillna(event_means)
df["has_prior_season_data"] = df["has_prior_season_data"].fillna(0).astype(int)

df = df.reset_index(drop=True)
event_dummies = pd.get_dummies(df["Event"], prefix="Event", dtype=int)
df = pd.concat([df, event_dummies], axis=1)

feature_cols = ["Week", "Year", "High_Score", "Average", "score_1", "score_2",
                "Event_BB", "Event_FX", "Event_UB", "Event_VT",
                "career_high", "prev_season_avg", "has_prior_season_data"]
target = "score_adj"

# =============================================================================
# Hyperparameter tuning using rolling-origin CV splits
# =============================================================================

# Build rolling-origin CV splits as positional index arrays
unique_weeks = sorted(df["Week"].unique())
cv_splits = []
for week in unique_weeks[2:]:   # need at least 2 weeks of training data
    train_idx = df.index[df["Week"] < week].tolist()
    test_idx  = df.index[df["Week"] == week].tolist()
    if train_idx and test_idx:
        cv_splits.append((train_idx, test_idx))

params = {
    "max_depth":        [3, 4, 5, 6],
    "min_child_weight": [5, 10, 15, 20],
    "subsample":        [0.5, 0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "gamma":            [0, 0.05, 0.1, 0.25, 0.5],
    "learning_rate":    [0.02, 0.05, 0.1],
    "n_estimators":     [300, 500],
}

xgb_base = xgb.XGBRegressor(objective="reg:squarederror", verbosity=0, nthread=1)
random_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=params,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=cv_splits,
    n_jobs=1,
    random_state=42,
    verbose=1,
)
random_search.fit(df[feature_cols], df[target])

print(f"\nBest params: {random_search.best_params_}")
print(f"Best rolling-origin CV RMSE: {-random_search.best_score_:.3f}")

# =============================================================================
# Rolling-origin evaluation with best params
# =============================================================================

results = []
for week in unique_weeks[1:]:
    train = df[df["Week"] < week]
    test  = df[df["Week"] == week]
    if train.empty or test.empty:
        continue
    model = xgb.XGBRegressor(**random_search.best_params_, objective="reg:squarederror", verbosity=0, nthread=1)
    model.fit(train[feature_cols], train[target])
    preds = model.predict(test[feature_cols])
    rmse = np.sqrt(mean_squared_error(test[target], preds))
    results.append({"week": week, "n": len(test), "rmse": round(rmse, 3)})

results_df = pd.DataFrame(results)
print("\nXGBoost rolling-origin evaluation (2022-2025, tuned):")
print(results_df.to_string(index=False))
print(f"Mean RMSE: {results_df['rmse'].mean():.3f}")
print("(Linear model mean RMSE 2022-2025 rolling-origin: 0.230)")

# =============================================================================
# Train final model on all 2022-2025 data; evaluate on held-out 2026 season
# =============================================================================

final_model = xgb.XGBRegressor(**random_search.best_params_, objective="reg:squarederror", verbosity=0, nthread=1)
final_model.fit(df[feature_cols], df[target])

df_2026 = pd.read_csv("../Files/scores_long_adjusted.csv")
df_2026 = df_2026[df_2026["Event"] != "AA"]
df_2026["Date"] = pd.to_datetime(df_2026["Date"], errors="coerce")
df_2026["Year"] = df_2026["Date"].dt.year.max()

df_2026 = df_2026.reset_index(drop=True)
event_dummies_26 = pd.get_dummies(df_2026["Event"], prefix="Event", dtype=int)
df_2026 = pd.concat([df_2026, event_dummies_26], axis=1)

df_2026 = df_2026.sort_values(["GymnastID", "Event", "Week"])
year_index_26 = df_2026["Year"]
event_index_26 = df_2026["Event"]
gymnast_index_26 = df_2026["GymnastID"]
df_2026 = df_2026.groupby(["GymnastID", "Event"], group_keys=False).apply(compute_rolling_features_linear, include_groups=False)
df_2026["Year"] = year_index_26
df_2026["Event"] = event_index_26
df_2026["GymnastID"] = gymnast_index_26
df_2026 = df_2026.dropna(subset=["score_2"])
df_2026 = df_2026.rename(columns={"high_score": "High_Score", "average": "Average"})

# Add prior season features for 2026 (using 2022-2025 data)
psf_2026 = compute_prior_season_features(df_all, 2026)
df_2026 = df_2026.merge(psf_2026, on=["GymnastID", "Event"], how="left")
for col in ["career_high", "prev_season_avg"]:
    event_means = df_2026.groupby("Event")[col].transform("mean")
    df_2026[col] = df_2026[col].fillna(event_means)
df_2026["has_prior_season_data"] = df_2026["has_prior_season_data"].fillna(0).astype(int)

results_2026 = []
for week in sorted(df_2026["Week"].unique()):
    test = df_2026[df_2026["Week"] == week]
    if test.empty:
        continue
    preds = final_model.predict(test[feature_cols])
    rmse = np.sqrt(mean_squared_error(test["score_adj"], preds))
    results_2026.append({"week": week, "n": len(test), "rmse_xgb": round(rmse, 3)})

res_2026 = pd.DataFrame(results_2026)
print("\nComparison on held-out 2026 data:")
print(res_2026.to_string(index=False))
print(f"\nMean RMSE XGBoost (2026): {res_2026['rmse_xgb'].mean():.3f}")
print("Mean RMSE Linear  (2026): 0.230")

# =============================================================================
# Feature importances
# =============================================================================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "gain":    final_model.get_booster().get_score(importance_type="gain").values() if final_model.get_booster().get_score(importance_type="gain") else [0]*len(feature_cols),
})

gain_scores = final_model.get_booster().get_score(importance_type="gain")
cover_scores = final_model.get_booster().get_score(importance_type="cover")
freq_scores = final_model.get_booster().get_score(importance_type="weight")

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "gain":    [round(gain_scores.get(f, 0), 3) for f in feature_cols],
    "cover":   [round(cover_scores.get(f, 0), 3) for f in feature_cols],
    "freq":    [round(freq_scores.get(f, 0), 3) for f in feature_cols],
}).sort_values("gain", ascending=False)

print("\nXGBoost feature importances (final model trained on 2022-2025):")
print("  gain  = avg improvement in loss when feature is used in a split")
print("  cover = avg number of samples affected by splits on this feature")
print("  freq  = number of times feature appears in trees")
print(importance_df.to_string(index=False))

# =============================================================================
# SHAP summary plot
# =============================================================================

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(df[feature_cols])

shap.summary_plot(shap_values, df[feature_cols], show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSHAP summary plot saved to shap_summary.png")

# =============================================================================
# First tree structure
# =============================================================================

booster = final_model.get_booster()
booster.feature_names = feature_cols
print("\nFirst tree structure:")
print(booster.get_dump(with_stats=True)[0])