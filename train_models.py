#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib


# 📌 Load the 2025 dataset for training
# one row per score
# home scores already adjusted based on league homeaway factor
df = pd.read_csv("2025 files/scores_long_adjusted.csv")

# 📌 Compute features
# One-Hot Encode the 'Event' column with all categories present
event_dummies = pd.get_dummies(df['Event'], prefix='Event', dtype=int)

# Add event dummies to the dataset
df = df.join(event_dummies)

# Sort by GymnastID, Event, and Date to ensure chronological order
df = df.sort_values(by=["GymnastID", "Event", "Date"])

# Function to compute rolling statistics up to each row's date
def compute_rolling_features(group):
    group["high_score"] = group["score_adj"].expanding().max().shift(1)
    group["low_score"] = group["score_adj"].expanding().min().shift(1)
    group["average"] = group["score_adj"].expanding().mean().shift(1)
    #group["last_3"] = (
    #    group["score_adj"].shift(1) + group["score_adj"].shift(2) + group["score_adj"].shift(3)
    #)/3
    group["score_1"] = group["score_adj"].shift(1)
    group["score_2"] = group["score_adj"].shift(2)
    #group["score_3"] = group["score_adj"].shift(3)

    # Calculate features to predict how likely they are to compete next
    # Rough metric - does not account for whether the team competed
    #group["competed_last_week"] = (
    #    group["Week"] - group["Week"].shift(1) <= 1
    #).astype(int)
    #group["prior_competitions"] = np.arange(len(group))
    #group["prior_competitions_percent"] = np.arange(len(group)) / group["Week"].astype(int)

    return group

# Apply function grouped by Gymnast and Event
df = df.groupby(["GymnastID", "Event"], group_keys=False).apply(compute_rolling_features)

# Require at least 2 prior scores
df = df.dropna(subset=["score_2"])

df.to_csv("linear_features.csv", index=False)

df

#%%
# 📌 Train and evaluate linear regression models based on rolling origin splits
from sklearn.metrics import mean_squared_error

# Define X (features) and Y (target)
features = ['Week','high_score','average','score_1','score_2', #'score_3',
            'Event_BB','Event_FX','Event_UB','Event_VT']
target = 'score_adj'

# Create empty lists to hold the results/metrics
results = []
coefs_list = []   # coefficients per week

# This should start at 3 
unique_weeks = sorted(df['Week'].unique())

for week in unique_weeks[2:]: # skip weeks 3-4 because not enough training data
    # define train and test sets
    train = df[df['Week'] < week]
    test  = df[df['Week'] == week]
    
    if test.empty:
        continue

    if train.empty:
        continue
    
    X_train = train[features]
    y_train = train[target]
    
    X_test  = test[features]
    y_test  = test[target]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'week': week,
        'rmse': round(rmse, 3),
        'n_test': len(test),
        'n_train': len(train)
    })
    
    # Save coefficients explicitly as separate columns
    coef_dict = {'week': int(week), 'intercept': model.intercept_}
    
    # Assign each coefficient to its corresponding feature
    for i, f in enumerate(features):
        coef_dict[f'coef_{f}'] = model.coef_[i]
    
    coefs_list.append(coef_dict)

# Convert to DataFrames
results_df = pd.DataFrame(results)
coefs_df   = pd.DataFrame(coefs_list)

print(results_df)
print(coefs_df)

#%%
from sklearn.linear_model import LinearRegression

# Train on all 2025 data
X = df[features]
y = df["score_adj"]

score_model = LinearRegression()
score_model.fit(X, y)
joblib.dump(score_model, "predict_score.joblib")

print(f"Model trained on {len(X)} rows of 2025 data")

#import matplotlib.pyplot as plt
#plt.plot(pred_df["pred_score"][pred_df["score_adj"]>9.0],pred_df["score_adj"][pred_df["score_adj"]>9.0],'o')
#plt.plot([9.5, 10.0], [9.5, 10.0], linestyle="--", color="gray", label="Perfect")
#plt.show()

#%%
# 📌 Functions for "Likelihood to Compete" model

def create_weekly_format(df):
    """Convert scores dataframe to weekly format with all gymnast/event/week combinations."""
    weekly_counts = (
        df
        .groupby(["GymnastID", "Event", "Week"])
        .size()
        .reset_index(name="n_competes")
    )

    # All rows will have competed_this_week = 1 if they competed at least once in a week
    weekly_counts["competed_this_week"] = (weekly_counts["n_competes"] > 0).astype(int)

    # Get all possible weeks
    gymnasts = df["GymnastID"].unique()
    events = df["Event"].unique()
    all_weeks = np.arange(1, df["Week"].max() + 1)

    # Create full MultiIndex for all combinations
    full_index = pd.MultiIndex.from_product(
        [gymnasts, events, all_weeks],
        names=["GymnastID", "Event", "Week"]
    )

    # Reindex weekly counts to this full index, add 0 if they did not compete
    weekly_full = (
        weekly_counts
        .set_index(["GymnastID", "Event", "Week"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Define competed_this_week
    weekly_full["competed_this_week"] = (weekly_full["n_competes"] > 0).astype(int)
    weekly_full = weekly_full.drop(columns="n_competes")

    return weekly_full


def compute_compete_features(weekly_full, player_info):
    """Compute features for likelihood to compete model."""
    # Join player info
    weekly_full = weekly_full.join(
        player_info.set_index('GymnastID'),
        on='GymnastID'
    )

    # Which teams competed each week
    weekly_team = (
        weekly_full
        .groupby(['Team', 'Week'])['competed_this_week']
        .max()
        .reset_index()
    )

    weekly_full = weekly_full.join(
        weekly_team.set_index(['Team', 'Week']),
        on=['Team', 'Week'],
        rsuffix='_team'
    )

    # Compute prior_competitions based on competed_this_week
    weekly_full = weekly_full.sort_values(["GymnastID", "Event", "Week"])

    weekly_full["prior_competitions_temp"] = (
        weekly_full
        .groupby(["GymnastID", "Event"])["competed_this_week"]
        .cumsum()
        .fillna(0)
        .astype(int)
    )

    weekly_full["team_competitions_temp"] = (
        weekly_full
        .groupby(["GymnastID", "Event"])["competed_this_week_team"]
        .cumsum()
        .fillna(0)
        .astype(int)
    )

    weekly_full["prior_competitions"] = weekly_full["prior_competitions_temp"] - weekly_full["competed_this_week"]
    weekly_full["team_competitions"] = weekly_full["team_competitions_temp"] - weekly_full["competed_this_week_team"]
    weekly_full = weekly_full.drop(columns=["prior_competitions_temp", "team_competitions_temp"])

    # Compute prior_competitions_percent
    weekly_full["prior_competitions_percent"] = (
        weekly_full["prior_competitions"] / weekly_full["team_competitions"]
    )
    weekly_full['prior_competitions_percent'] = weekly_full['prior_competitions_percent'].fillna(0.0)

    # Define competed_last_week
    weekly_full["competed_last_week"] = (
        weekly_full
        .groupby(["GymnastID", "Event"])["competed_this_week"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return weekly_full

#%%
# 📌 Create weekly format for 2025 data

df = pd.read_csv("2025 files/scores_long_adjusted.csv")
info = pd.read_csv("2025 files/player_info.csv")

weekly_full = create_weekly_format(df)
weekly_full = compute_compete_features(weekly_full, info)

weekly_full
# weekly_full.query("GymnastID == 33386 & Event == 'BB'")

#%%
# 📌 Train and evaluate "Likelihood to Compete" logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

# Define X (features) and Y (target)
log_features = ['Week', 'prior_competitions_percent', 'competed_last_week']
log_target = 'competed_this_week'

log_results = []
log_coefs_list = []   # coefficients per week

unique_weeks = sorted(weekly_full['Week'].unique())

for week in unique_weeks[2:]: # start at week 3 - more training data
    # define train and test sets
    train = weekly_full[weekly_full['Week'] < week]
    test  = weekly_full[weekly_full['Week'] == week]
    
    if test.empty:
        continue

    if train.empty:
        continue
    
    X_train = train[log_features]
    y_train = train[log_target]
    
    X_test  = test[log_features]
    y_test  = test[log_target]
    
    model = LogisticRegression(
        class_weight = "balanced",
        penalty="l2", # Ridge - penalizes large coef, stabilizes them
        C=1.0, # inverse regularization, 1.0 is weak (low C/high reg prevents coef from exploding)
        max_iter=1000 # prevents errors/failing to converge
    )

    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]

    ll  = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    log_results.append({
        'week': week,
        'log_loss': round(ll, 4),
        'auc': round(auc, 3),
        'n_test': len(test),
        'n_train': len(train),
        'event_rate_test': y_test.mean()
    })

    coef_dict = {'week': int(week), 'intercept': model.intercept_[0]}

    for i, f in enumerate(log_features):
        coef_dict[f'coef_{f}'] = model.coef_[0, i]

    log_coefs_list.append(coef_dict)


# Convert to DataFrames
log_results_df = pd.DataFrame(log_results)
log_coefs_df   = pd.DataFrame(log_coefs_list)

print(log_results_df)
print(log_coefs_df)

#%%
# Plot calibration curve

cal_df = pd.DataFrame({
    "y_true": y_test.values.ravel(),
    "y_prob": y_prob
})

# Create probability deciles (10 equal-sized bins)
cal_df["prob_decile"] = pd.qcut(
    cal_df["y_prob"],
    q=10,
    duplicates="drop"
)

# Aggregate observed vs predicted
cal_summary = (
    cal_df
    .groupby("prob_decile")
    .agg(
        mean_pred_prob=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
        n=("y_true", "size")
    )
    .reset_index()
)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

plt.plot(
    cal_summary["mean_pred_prob"],
    cal_summary["observed_rate"],
    marker="o",
    label="Model"
)

# Perfect calibration line
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

plt.xlabel("Predicted probability")
plt.ylabel("Observed competition rate")
plt.title("Calibration: Predicted vs Observed")
plt.legend()
plt.grid(True)

plt.show()

# Interpretation:
## bins collapsed from 10 to 6 - most grouped at very low probability but that's expected
## Tends to under-predict - that's okay as long as ranking is good (based on AOC it is)
### Just note that it tends to be pessimistic, seems like .5 is a good cutoff

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

X = weekly_full[log_features]         # Week, prior_competitions_percent, competed_last_week
y = weekly_full["competed_this_week"]

model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
model.fit(X, y)
joblib.dump(model, "likelihood_to_compete.joblib")

model.coef_

#%%
# Plot calibration curve for training data
train_prob = model.predict_proba(X)[:, 1]

cal_df = pd.DataFrame({
    "y_true": y.values.ravel(),
    "y_prob": train_prob
})

# Create probability deciles (10 equal-sized bins)
cal_df["prob_decile"] = pd.qcut(
    cal_df["y_prob"],
    q=10,
    duplicates="drop"
)

# Aggregate observed vs predicted
cal_summary = (
    cal_df
    .groupby("prob_decile")
    .agg(
        mean_pred_prob=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
        n=("y_true", "size")
    )
    .reset_index()
)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

plt.plot(
    cal_summary["mean_pred_prob"],
    cal_summary["observed_rate"],
    marker="o",
    label="Model"
)

# Perfect calibration line
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

plt.xlabel("Predicted probability")
plt.ylabel("Observed competition rate")
plt.title("Calibration: Predicted vs Observed")
plt.legend()
plt.grid(True)

plt.show()