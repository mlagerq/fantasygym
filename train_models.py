#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib


# Function to compute rolling statistics up to each row's date
def compute_rolling_features(group):
    group["high_score"] = group["score_adj"].expanding().max().shift(1)
    group["low_score"] = group["score_adj"].expanding().min().shift(1)
    group["average"] = group["score_adj"].expanding().mean().shift(1)
    group["score_1"] = group["score_adj"].shift(1)
    group["score_2"] = group["score_adj"].shift(2)
    return group


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


# ==============================================================================
# Training code - only runs when executed directly, not when imported
# ==============================================================================

if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    # =========================================================================
    # Train Score Prediction Model
    # =========================================================================

    # Load the 2025 dataset for training
    df = pd.read_csv("2025 files/scores_long_adjusted.csv")

    # One-Hot Encode the 'Event' column
    event_dummies = pd.get_dummies(df['Event'], prefix='Event', dtype=int)
    df = df.join(event_dummies)

    # Sort by GymnastID, Event, and Date to ensure chronological order
    df = df.sort_values(by=["GymnastID", "Event", "Date"])

    # Apply function grouped by Gymnast and Event
    df = df.groupby(["GymnastID", "Event"], group_keys=False).apply(compute_rolling_features)

    # Require at least 2 prior scores
    df = df.dropna(subset=["score_2"])
    df.to_csv("linear_features.csv", index=False)

    # Define features and target
    features = ['Week','high_score','average','score_1','score_2',
                'Event_BB','Event_FX','Event_UB','Event_VT']
    target = 'score_adj'

    # Evaluate with rolling origin
    results = []
    unique_weeks = sorted(df['Week'].unique())

    for week in unique_weeks[2:]:
        train = df[df['Week'] < week]
        test = df[df['Week'] == week]

        if test.empty or train.empty:
            continue

        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({'week': week, 'rmse': round(rmse, 3), 'n_test': len(test)})

    print("Score Model Evaluation:")
    print(pd.DataFrame(results))

    # Train final model on all data
    X = df[features]
    y = df["score_adj"]
    score_model = LinearRegression()
    score_model.fit(X, y)
    joblib.dump(score_model, "predict_score.joblib")
    print(f"Score model trained on {len(X)} rows")

    # =========================================================================
    # Train Likelihood to Compete Model
    # =========================================================================

    df = pd.read_csv("2025 files/scores_long_adjusted.csv")
    info = pd.read_csv("2025 files/player_info.csv")

    weekly_full = create_weekly_format(df)
    weekly_full = compute_compete_features(weekly_full, info)

    log_features = ['Week', 'prior_competitions_percent', 'competed_last_week']
    log_target = 'competed_this_week'

    # Evaluate with rolling origin
    log_results = []
    unique_weeks = sorted(weekly_full['Week'].unique())

    for week in unique_weeks[2:]:
        train = weekly_full[weekly_full['Week'] < week]
        test = weekly_full[weekly_full['Week'] == week]

        if test.empty or train.empty:
            continue

        X_train, y_train = train[log_features], train[log_target]
        X_test, y_test = test[log_features], test[log_target]

        model = LogisticRegression(class_weight="balanced", penalty="l2", C=1.0, max_iter=1000)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        log_results.append({'week': week, 'auc': round(auc, 3), 'n_test': len(test)})

    print("\nLikelihood Model Evaluation:")
    print(pd.DataFrame(log_results))

    # Train final model on all data
    X = weekly_full[log_features]
    y = weekly_full["competed_this_week"]
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, "likelihood_to_compete.joblib")
    print(f"Likelihood model trained on {len(X)} rows")
