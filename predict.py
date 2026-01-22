#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
from train_models import create_weekly_format, compute_compete_features, compute_rolling_features


def run_predictions(scores_csv="scores_long_adjusted.csv", output_csv="team_opt_input_linear.csv"):
    """
    Run score and likelihood predictions for the next week.

    Args:
        scores_csv: Path to adjusted scores data
        output_csv: Path to save combined predictions with pricing

    Returns:
        DataFrame with predictions ready for team optimizer
    """
    # Load trained models
    score_model = joblib.load("predict_score.joblib")
    compete_model = joblib.load("likelihood_to_compete.joblib")

    # Define features for score prediction
    features = ['Week','high_score','average','score_1','score_2',
                'Event_BB','Event_FX','Event_UB','Event_VT']
    log_features = ['Week', 'prior_competitions_percent', 'competed_last_week']

    # =========================================================================
    # PART 1: Score Predictions
    # =========================================================================

    # Load current season data
    df_2026 = pd.read_csv(scores_csv)

    # Create blank next week rows for each gymnast/event combination
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

    print(f"Score predictions made for {len(pred_df_2026)} gymnast-events for week {next_week}")

    pred_df_2026.to_csv(f'predictions_week_{next_week}.csv', index=False)

    # =========================================================================
    # PART 2: Likelihood to Compete Predictions
    # =========================================================================

    # Create weekly format for current data
    df = pd.read_csv(scores_csv)
    info = pd.read_csv("player_info.csv")

    weekly_full = create_weekly_format(df)
    weekly_full = compute_compete_features(weekly_full, info)

    # Create data for next week prediction
    next_week = weekly_full["Week"].max() + 1
    gymnasts = weekly_full["GymnastID"].unique()
    events = weekly_full["Event"].unique()

    # Create full index for next week
    next_week_index = pd.MultiIndex.from_product(
        [gymnasts, events],
        names=["GymnastID", "Event"]
    )

    # Build DataFrame
    next_week_df = pd.DataFrame(index=next_week_index).reset_index()
    next_week_df["Week"] = next_week

    # Compute features: prior competitions percent
    prior_counts = (
        weekly_full
        .groupby(["GymnastID", "Event"])["competed_this_week"]
        .sum()
        .rename("prior_competitions")
    )
    next_week_df = next_week_df.merge(prior_counts, on=["GymnastID", "Event"], how="left")
    next_week_df["prior_competitions"].fillna(0, inplace=True)
    next_week_df["prior_competitions_percent"] = next_week_df["prior_competitions"] / (next_week - 1)
    next_week_df = next_week_df.drop(columns="prior_competitions")

    # Competed last week
    last_week = weekly_full[weekly_full["Week"] == weekly_full["Week"].max()]
    last_week_flag = last_week.set_index(["GymnastID", "Event"])["competed_this_week"].rename("competed_last_week")
    next_week_df = next_week_df.merge(last_week_flag, on=["GymnastID", "Event"], how="left")
    next_week_df["competed_last_week"].fillna(0, inplace=True)

    # Ensure correct type
    next_week_df[["prior_competitions_percent","competed_last_week"]] = next_week_df[["prior_competitions_percent","competed_last_week"]].astype(float)

    # Predict likelihood to compete
    X_next = next_week_df[log_features]
    next_week_df["pred_prob"] = compete_model.predict_proba(X_next)[:, 1]

    print(f"Likelihood predictions made for {len(next_week_df)} gymnast-events")

    # =========================================================================
    # PART 3: Combine with Pricing
    # =========================================================================

    pricing = pd.read_csv("fantasizr_player_pricing.csv")
    info = pd.read_csv("player_info.csv")

    pricing = pricing.join(
        info.set_index('Name'),
        on='Player Name',
        lsuffix='x'
    )

    all_pred = pd.merge(
        pred_df_2026[["GymnastID", "Event", "pred_score"]],
        next_week_df[['GymnastID', 'Event', 'pred_prob']],
        left_on=['GymnastID', 'Event'],
        right_on=['GymnastID', 'Event'],
        how='inner'
    )
    final_df = pd.merge(
        all_pred,
        pricing[['GymnastID', 'Player Name', 'Team', 'Event', 'Price']],
        left_on=['GymnastID', 'Event'],
        right_on=['GymnastID', 'Event'],
        how='inner'
    )
    final_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} with {len(final_df)} rows")

    return final_df


if __name__ == "__main__":
    run_predictions()
