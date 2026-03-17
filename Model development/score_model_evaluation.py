#%%
"""
Score Model Evaluation
======================
Compares score prediction models trained on different datasets,
evaluated on held-out 2026 season data.

Models compared:
  - 2025-only:   trained on 2025 season, features without Year
  - 2022-2025:   trained on 2022-2025 seasons, adds Year as a continuous
                 feature to capture inter-season scoring inflation

Key findings:
  - 2022-2025 model outperforms 2025-only on every week of 2026 data
  - Mean RMSE: 0.230 (2022-2025) vs 0.233 (2025-only)
  - Year coefficient ~0.005: each season adds ~0.005 pts to predicted score
  - Event coefficients become small and meaningful once AA rows excluded from training
    (previously ~-3.9 due to AA scores inflating the intercept)

Other decisions documented here:
  - AA excluded from training (predicted as sum of 4 events, not directly modelled)
  - Weeks 13+ excluded (post-season dynamics differ from regular season)
  - Rolling features grouped by GymnastID + Event + Year so they reset each season
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_models import compute_rolling_features


# =============================================================================
# Prepare 2026 test data
# =============================================================================

df_2026 = pd.read_csv("Files/scores_long_adjusted.csv")
df_2026 = df_2026[df_2026["Event"] != "AA"]
df_2026["Year"] = pd.to_datetime(df_2026["Date"], errors="coerce").dt.year.max()
event_dummies = pd.get_dummies(df_2026["Event"], prefix="Event", dtype=int)
df_2026 = df_2026.join(event_dummies)

# Group rolling features by gymnast+season so they reset each year
df_2026["GymnastSeason"] = df_2026["GymnastID"].astype(str) + "_" + df_2026["Year"].astype(str)
df_2026 = df_2026.sort_values(["GymnastSeason", "Event", "Week"])
year_index = df_2026["Year"]
df_2026 = df_2026.groupby(["GymnastSeason", "Event"], group_keys=False).apply(
    compute_rolling_features, include_groups=False
)
df_2026["Year"] = year_index
df_2026 = df_2026.dropna(subset=["score_2"])

print(f"2026 test weeks: {sorted(df_2026['Week'].unique())}")
print(f"2026 test rows:  {len(df_2026)}")

# =============================================================================
# Train 2025-only baseline model
# =============================================================================

df_2025 = pd.read_csv("2025 files/scores_long_adjusted.csv")
df_2025 = df_2025[df_2025["Event"] != "AA"]
event_dummies_25 = pd.get_dummies(df_2025["Event"], prefix="Event", dtype=int)
df_2025 = df_2025.join(event_dummies_25)
df_2025 = df_2025.sort_values(["GymnastID", "Event", "Date"])
df_2025 = df_2025.groupby(["GymnastID", "Event"], group_keys=False).apply(
    compute_rolling_features, include_groups=False
)
df_2025 = df_2025.dropna(subset=["score_2"])

features_old = ["Week", "high_score", "average", "score_1", "score_2",
                "Event_BB", "Event_FX", "Event_UB", "Event_VT"]
features_new = ["Week", "Year", "high_score", "average", "score_1", "score_2",
                "Event_BB", "Event_FX", "Event_UB", "Event_VT"]

model_2025 = LinearRegression()
model_2025.fit(df_2025[features_old], df_2025["score_adj"])
print(f"\n2025-only model trained on {len(df_2025)} rows")

# =============================================================================
# Load 2022-2025 model (trained by train_models.py)
# =============================================================================

model_multi = joblib.load("predict_score.joblib")
print(f"2022-2025 model loaded")

print("\n2022-2025 model coefficients:")
for feat, coef in zip(model_multi.feature_names_in_, model_multi.coef_):
    print(f"  {feat}: {coef:.6f}")

# =============================================================================
# Evaluate both models on 2026 data (week by week)
# =============================================================================

results = []
for week in sorted(df_2026["Week"].unique()):
    test = df_2026[df_2026["Week"] == week]
    if test.empty:
        continue
    rmse_2025 = np.sqrt(mean_squared_error(test["score_adj"], model_2025.predict(test[features_old])))
    rmse_multi = np.sqrt(mean_squared_error(test["score_adj"], model_multi.predict(test[features_new])))
    results.append({
        "week": week,
        "n": len(test),
        "rmse_2025only": round(rmse_2025, 3),
        "rmse_2022_2025": round(rmse_multi, 3),
    })

res = pd.DataFrame(results)
print("\nEvaluation on 2026 held-out data:")
print(res.to_string(index=False))
print(f"\nMean RMSE  2025-only:  {res['rmse_2025only'].mean():.3f}")
print(f"Mean RMSE  2022-2025:  {res['rmse_2022_2025'].mean():.3f}")
