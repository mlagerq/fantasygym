#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
from train_models import create_weekly_format, compute_compete_features, compute_rolling_features

# Load trained models
score_model = joblib.load("predict_score.joblib")

# Define features for score prediction
features = ['Week','high_score','average','score_1','score_2',
            'Event_BB','Event_FX','Event_UB','Event_VT']

# 📌 Predict week 4 of 2026 using 2025-trained model

# Load 2026 data
df_2026 = pd.read_csv("scores_long_adjusted.csv")

# Create blank week 4 rows for each gymnast/event combination
gymnast_events = df_2026[["GymnastID", "Event"]].drop_duplicates()
next_week = df_2026["Week"].max() + 1
next_week_rows = gymnast_events.copy()
next_week_rows["Week"] = next_week

# Append blank next week rows to the data
df_2026 = pd.concat([df_2026, next_week_rows], ignore_index=True)

# Add event dummies
event_dummies_2026 = pd.get_dummies(df_2026['Event'], prefix='Event', dtype=int)
df_2026 = df_2026.join(event_dummies_2026)

# Sort and compute rolling features (next week rows will get features from prior weeks)
df_2026 = df_2026.sort_values(by=["GymnastID", "Event", "Week"])
df_2026 = df_2026.groupby(["GymnastID", "Event"], group_keys=False).apply(compute_rolling_features)

# Extract next week rows and filter to those with 2+ prior scores
pred_df_2026 = df_2026[df_2026["Week"] == next_week].copy()
pred_df_2026 = pred_df_2026.dropna(subset=["score_2"])

# Predict only for individual events (not AA)
pred_df_2026 = pred_df_2026[pred_df_2026["Event"] != "AA"]
X_2026 = pred_df_2026[features]
pred_df_2026["pred_score"] = score_model.predict(X_2026)

# Create AA predictions by summing 4 event predictions per gymnast
aa_preds = pred_df_2026.groupby("GymnastID").agg({
    "pred_score": "sum",
    "Week": "first"
}).reset_index()
aa_preds["Event"] = "AA"

# Append AA predictions to the main predictions
pred_df_2026 = pd.concat([pred_df_2026, aa_preds], ignore_index=True)

print(f"2026 Predictions made for {len(pred_df_2026)} rows for week_" + str(next_week))

pred_df_2026.to_csv('predictions_week_' + str(next_week) + '.csv',index=False)

#%%
# 📌 Create weekly format for current data

df = pd.read_csv("scores_long_adjusted.csv")
info = pd.read_csv("player_info.csv")

weekly_full = create_weekly_format(df)
weekly_full = compute_compete_features(weekly_full, info)

weekly_full
#%%
# Create data to send to model for next week 

# 1️⃣ Define next week
next_week = weekly_full["Week"].max() + 1

# 2️⃣ Get all gymnast-event combinations
gymnasts = weekly_full["GymnastID"].unique()
events    = weekly_full["Event"].unique()

# 3️⃣ Create full index for next week
next_week_index = pd.MultiIndex.from_product(
    [gymnasts, events],
    names=["GymnastID", "Event"]
)

# 4️⃣ Build DataFrame
next_week_df = pd.DataFrame(index=next_week_index).reset_index()
next_week_df["Week"] = next_week

# 5️⃣ Compute features

# prior competitions count up to last week
prior_counts = (
    weekly_full
    .groupby(["GymnastID", "Event"])["competed_this_week"]
    .sum()
    .rename("prior_competitions")
)
next_week_df = next_week_df.merge(prior_counts, on=["GymnastID", "Event"], how="left")
next_week_df["prior_competitions"].fillna(0, inplace=True)

# prior competitions percent (relative to number of past weeks)
next_week_df["prior_competitions_percent"] = next_week_df["prior_competitions"] / (next_week - 1)
next_week_df = next_week_df.drop(columns="prior_competitions")

# competed last week
last_week = weekly_full[weekly_full["Week"] == weekly_full["Week"].max()]
last_week_flag = last_week.set_index(["GymnastID", "Event"])["competed_this_week"].rename("competed_last_week")
next_week_df = next_week_df.merge(last_week_flag, on=["GymnastID", "Event"], how="left")
next_week_df["competed_last_week"].fillna(0, inplace=True)

# Ensure correct type
next_week_df[["prior_competitions_percent","competed_last_week"]] = next_week_df[["prior_competitions_percent","competed_last_week"]].astype(float)

# 6️⃣ Ready to predict
X_next = next_week_df[log_features]  # features = ["Week","prior_competitions_percent","competed_last_week"]

next_week_df
#%%
# Predict likelihood to compete next week
from sklearn.linear_model import LogisticRegression

# Load trained model
model = joblib.load("likelihood_to_compete.joblib")

# Predict probabilities
next_week_df["pred_prob"] = model.predict_proba(X_next)[:, 1]

#%%
# Plot distribution of pred_prob
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(next_week_df["pred_prob"], bins=30, edgecolor='black')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Likelihood to Compete (pred_prob)")
plt.show()

#%%
# Combine all the info into one dataset
pricing = pd.read_csv("fantasizr_player_pricing.csv")
info = pd.read_csv("player_info.csv")

pricing = pricing.join(
    info.set_index('Name'),
    on = 'Player Name',
    lsuffix = 'x'
)

all_pred = pd.merge(
    pred_df_2026[["GymnastID", "Event",  "pred_score"]],
    next_week_df[['GymnastID','Event','pred_prob']],
    left_on=['GymnastID', 'Event'],
    right_on=['GymnastID', 'Event'],
    how='inner'
)
final_df = pd.merge(
    all_pred,
    pricing[['GymnastID','Player Name','Team','Event','Price']], 
    left_on=['GymnastID', 'Event'], 
    right_on=['GymnastID', 'Event'], 
    how='inner'
)
final_df.to_csv("team_opt_input_linear.csv",index=False)
