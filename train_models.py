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
    """Convert scores dataframe to weekly format with all gymnast/event/week combinations.
    If a Year column is present, the grid is created independently per year."""
    if 'Year' in df.columns:
        frames = [_create_weekly_format_single(year_df).assign(Year=year)
                  for year, year_df in df.groupby('Year')]
        return pd.concat(frames, ignore_index=True)
    return _create_weekly_format_single(df)


def _create_weekly_format_single(df):
    """Create weekly format for a single season."""
    weekly_counts = (
        df
        .groupby(["GymnastID", "Event", "Week"])
        .size()
        .reset_index(name="n_competes")
    )
    weekly_counts["competed_this_week"] = (weekly_counts["n_competes"] > 0).astype(int)

    gymnasts = df["GymnastID"].unique()
    events = df["Event"].unique()
    all_weeks = np.arange(1, df["Week"].max() + 1)

    full_index = pd.MultiIndex.from_product(
        [gymnasts, events, all_weeks],
        names=["GymnastID", "Event", "Week"]
    )

    weekly_full = (
        weekly_counts
        .set_index(["GymnastID", "Event", "Week"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    weekly_full["competed_this_week"] = (weekly_full["n_competes"] > 0).astype(int)
    weekly_full = weekly_full.drop(columns="n_competes")
    return weekly_full


def compute_compete_features(weekly_full, player_info):
    """Compute features for likelihood to compete model.
    If a Year column is present, cumulative counts reset each year."""
    multi_year = 'Year' in weekly_full.columns
    gymnast_group = ["GymnastID", "Event", "Year"] if multi_year else ["GymnastID", "Event"]
    team_group    = ["Team", "Week", "Year"]        if multi_year else ["Team", "Week"]

    # Join player info
    weekly_full = weekly_full.join(
        player_info.set_index('GymnastID'),
        on='GymnastID'
    )

    # Which teams competed each week
    weekly_team = (
        weekly_full
        .groupby(team_group)['competed_this_week']
        .max()
        .reset_index()
    )

    weekly_full = weekly_full.join(
        weekly_team.set_index(team_group),
        on=team_group,
        rsuffix='_team'
    )

    # Compute prior_competitions based on competed_this_week
    weekly_full = weekly_full.sort_values(gymnast_group + ["Week"])

    weekly_full["prior_competitions_temp"] = (
        weekly_full
        .groupby(gymnast_group)["competed_this_week"]
        .cumsum()
        .fillna(0)
        .astype(int)
    )

    weekly_full["team_competitions_temp"] = (
        weekly_full
        .groupby(gymnast_group)["competed_this_week_team"]
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

    # Define competed_last_week (0 at the start of each season)
    weekly_full["competed_last_week"] = (
        weekly_full
        .groupby(gymnast_group)["competed_this_week"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return weekly_full


def compute_prior_season_features(df, target_year):
    """
    Compute career_high, prev_season_avg, and has_prior_season_data for each
    GymnastID+Event combination, using data from years prior to target_year.

    Freshmen (no prior data) are imputed with per-event league means and
    has_prior_season_data=0.

    Args:
        df: Full multi-year scores DataFrame with Year column (all events, no AA filter needed)
        target_year: The season being predicted/trained

    Returns:
        DataFrame with columns: GymnastID, Event, career_high, prev_season_avg, has_prior_season_data
    """
    prior = df[(df["Year"] < target_year) & (df["Event"] != "AA")].copy()

    # Career high: best score ever before target_year
    career_high = (
        prior.groupby(["GymnastID", "Event"])["score_adj"]
        .max()
        .rename("career_high")
        .reset_index()
    )

    # Previous season average: mean score in target_year - 1
    prev_year_data = prior[prior["Year"] == target_year - 1]
    prev_season_avg = (
        prev_year_data.groupby(["GymnastID", "Event"])["score_adj"]
        .mean()
        .rename("prev_season_avg")
        .reset_index()
    )

    # Merge career high and prev season avg
    features = career_high.merge(prev_season_avg, on=["GymnastID", "Event"], how="outer")
    features["has_prior_season_data"] = features["career_high"].notna().astype(int)

    # League mean per event (for imputation of freshmen)
    league_means = (
        prior.groupby("Event")["score_adj"]
        .mean()
        .rename("league_mean")
        .reset_index()
    )

    features = features.merge(league_means, on="Event", how="left")
    features["career_high"] = features["career_high"].fillna(features["league_mean"])
    features["prev_season_avg"] = features["prev_season_avg"].fillna(features["league_mean"])
    features = features.drop(columns=["league_mean"])

    return features


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

    # Load combined 2022-2025 dataset for training
    df_all_years = pd.read_csv("Historical/scores_all_years_adjusted.csv")
    df = df_all_years.copy()

    # One-Hot Encode the 'Event' column
    event_dummies = pd.get_dummies(df['Event'], prefix='Event', dtype=int)
    df = df.join(event_dummies)

    # Exclude post-season weeks (regionals/nationals) — different dynamics, not predicted
    df = df[df['Week'] <= 12]

    # Exclude AA — predicted as sum of 4 events, not directly modelled
    df = df[df['Event'] != 'AA']

    # Compute prior season features (career_high, prev_season_avg, has_prior_season_data)
    # For each year in training data, use all data from prior years.
    # Must be done before the groupby.apply since include_groups=False drops the Event column.
    prior_season_frames = []
    for yr in sorted(df['Year'].unique()):
        psf = compute_prior_season_features(df_all_years, yr)
        psf['Year'] = yr
        prior_season_frames.append(psf)
    prior_season_df = pd.concat(prior_season_frames, ignore_index=True)
    df = df.merge(prior_season_df, on=['GymnastID', 'Event', 'Year'], how='left')

    # Impute any remaining NaN (e.g. 2022 has no prior years; use per-event mean from non-NaN rows)
    for col in ['career_high', 'prev_season_avg']:
        event_means = df.groupby('Event')[col].transform('mean')
        df[col] = df[col].fillna(event_means)
    df['has_prior_season_data'] = df['has_prior_season_data'].fillna(0).astype(int)

    # Sort by GymnastID, Event, Year, and Date to ensure chronological order
    df = df.sort_values(by=["GymnastID", "Event", "Year", "Date"])

    # Apply rolling features grouped by Gymnast+Season and Event so they reset each season.
    # Save Year first since include_groups=False drops groupby keys from output.
    df['GymnastSeason'] = df['GymnastID'].astype(str) + '_' + df['Year'].astype(str)
    year_index = df['Year']
    df = df.groupby(["GymnastSeason", "Event"], group_keys=False).apply(compute_rolling_features, include_groups=False)
    df['Year'] = year_index

    # Require at least 2 prior scores
    df = df.dropna(subset=["score_2"])
    df.to_csv("linear_features.csv", index=False)

    # Define features and target
    features = ['Week','Year','high_score','average','score_1','score_2',
                'Event_BB','Event_FX','Event_UB','Event_VT',
                'career_high','prev_season_avg','has_prior_season_data']
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

    df = pd.read_csv("Historical/scores_all_years_adjusted.csv")
    df = df[df['Week'] <= 12]
    info = pd.read_csv("Files/player_info.csv")

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
