#%%
import pandas as pd
import numpy as np


def _infer_week_1_start(df):
    """Find the Tuesday on or before the earliest competition date in df."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    first = df['Date'].min()
    days_since_tuesday = (first.weekday() - 1) % 7
    return first - pd.Timedelta(days=days_since_tuesday)


def clean_data(input_csv="Files/road_to_nationals.csv", week_1_start="2025-12-30",
               output_long_csv="Files/road_to_nationals_long.csv",
               output_adj_csv="Files/scores_long_adjusted.csv",
               output_player_info_csv="Files/player_info.csv"):
    """
    Clean and transform scraped Road to Nationals data.

    Args:
        input_csv: Path to raw scraped data
        week_1_start: Start date of week 1. Pass None to infer from data.
        output_long_csv: Path for long-format scores (None to skip)
        output_adj_csv: Path for adjusted scores
        output_player_info_csv: Path for player info (None to skip)

    Returns:
        DataFrame with cleaned, adjusted scores in long format
    """
    # Load the scraped data
    df = pd.read_csv(input_csv)

    ## Create dataframe player_info
    player_info = df[['GymnastID', 'Name', 'Team']].drop_duplicates()
    if output_player_info_csv:
        player_info.to_csv(output_player_info_csv, index=False)
        print(f"Saved {output_player_info_csv} with {len(player_info)} gymnasts")

    ## Use date to infer week of competition
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Define the start of the first week
    if week_1_start is None:
        week_1_start = _infer_week_1_start(df)
        print(f"Inferred week_1_start: {week_1_start.date()}")
    week_1_start = pd.to_datetime(week_1_start)

    # Calculate the week number
    df['Week'] = ((df['Date'] - week_1_start).dt.days // 7) + 1

    # Remove invalid dates (if any)
    df = df.dropna(subset=['Week'])

    # Convert 'Week' to integer type
    df['Week'] = df['Week'].astype(int)

    # Step 1: Create unique MeetID for hosts
    meet_df = df[df["HomeAway"] == "H"][["Date", "Team"]].drop_duplicates()

    # Generate MeetID
    meet_df["MeetID"] = meet_df["Date"].astype(str) + "_" + meet_df["Team"]

    # Step 2: Handle neutral site meets (no home team)
    # For each date without a home team, pick first team alphabetically as pseudo-host
    dates_with_home = set(meet_df["Date"].unique())
    neutral_meets = df[~df["Date"].isin(dates_with_home)].groupby("Date").agg(
        Team=pd.NamedAgg(column="Team", aggfunc=lambda x: sorted(x.unique())[0])
    ).reset_index()
    neutral_meets["MeetID"] = neutral_meets["Date"].astype(str) + "_" + neutral_meets["Team"]

    # Add neutral meets to meet_df
    meet_df = pd.concat([meet_df, neutral_meets], ignore_index=True)

    # Step 3: Merge MeetID for host teams (and pseudo-hosts)
    df = df.merge(meet_df, on=["Date", "Team"], how="left")

    # Step 4: Assign MeetID to away teams by checking if their Team is in the Opponent column
    def assign_meetid(row):
        if pd.isna(row["MeetID"]):
            # Find the meet ID for the matching date
            matching_meets = meet_df.loc[meet_df["Date"] == row["Date"], "MeetID"].values
            if len(matching_meets) > 0 and any(row["Team"] in opp for opp in df.loc[df["Date"] == row["Date"], "Opponent"].dropna()):
                return matching_meets[0]  # Assign the first valid MeetID found
        return row["MeetID"]

    df["MeetID"] = df.apply(assign_meetid, axis=1)

    # Transform scores df to have events on separate rows
    # Melt the DataFrame to make one row per event per week per meet
    df_melted = df.melt(id_vars=['GymnastID', 'Team', 'HomeAway', 'Week', 'Date', 'MeetID'],
                         value_vars=['VT', 'UB', 'BB', 'FX', 'AA'],
                         var_name='Event',
                         value_name='Score')

    # Save the reformatted DataFrame
    if output_long_csv:
        df_melted.to_csv(output_long_csv, index=False)
        print(f"Saved {output_long_csv} with {len(df_melted)} rows")

    # Load per-team homeaway factors, with league-wide fallback
    team_homeaway_factor = pd.read_csv("Files/team_homeaway_factor.csv")
    league_homeaway_factor = pd.read_csv("Files/league_homeaway_factor.csv")

    # Apply homeaway adjustment to scores
    df_adj = df_melted.dropna().copy()
    df_adj = df_adj.merge(
        team_homeaway_factor[['Team', 'Event', 'homeaway_factor_shrunk']],
        on=['Team', 'Event'],
        how='left'
    )

    # Fill missing team factors with league-wide fallback
    league_dict = dict(zip(league_homeaway_factor['Event'], league_homeaway_factor['homeaway_factor']))
    df_adj['homeaway_factor'] = df_adj.apply(
        lambda row: league_dict.get(row['Event'], 0) if pd.isna(row['homeaway_factor_shrunk']) else row['homeaway_factor_shrunk'],
        axis=1
    )

    # Subtract homeaway factor from home meet scores
    df_adj['score_adj'] = np.where(
        df_adj['HomeAway'] == 'H',
        df_adj['Score'] - df_adj['homeaway_factor'],
        df_adj['Score']
    )

    df_adj.to_csv(output_adj_csv, index=False)
    print(f"Saved {output_adj_csv} with {len(df_adj)} rows")

    return df_adj


def prepare_historical_data(
    year_csvs=None,
    output_dir="Historical"
):
    """
    Process raw historical scraped files into adjusted scores for model training.
    Week 1 start is inferred automatically as the Tuesday before the first competition.

    Args:
        year_csvs: dict of {year: raw_csv_path}. Defaults to 2022-2025.
        output_dir: folder to write scores_YYYY_adjusted.csv files

    Returns:
        Combined DataFrame with all years, including a Year column.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if year_csvs is None:
        year_csvs = {
            2022: "Historical/scores_2022.csv",
            2023: "Historical/scores_2023.csv",
            2024: "Historical/scores_2024.csv",
            2025: "2025 files/road_to_nationals.csv",
        }

    frames = []
    for year, path in year_csvs.items():
        print(f"\n--- Processing {year} ---")
        out_path = f"{output_dir}/scores_{year}_adjusted.csv"
        df_adj = clean_data(
            input_csv=path,
            week_1_start=None,           # infer from data
            output_long_csv=None,        # skip intermediate file
            output_adj_csv=out_path,
            output_player_info_csv=None, # skip player info
        )
        df_adj['Year'] = year
        frames.append(df_adj)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(f"{output_dir}/scores_all_years_adjusted.csv", index=False)
    print(f"\nSaved combined dataset: {len(combined)} rows across {combined['Year'].nunique()} years")
    return combined


# ==============================================================================
# ONE-TIME SETUP FUNCTIONS (run once to generate homeaway factor CSVs)
# ==============================================================================

def _raw_to_long(filepath, year):
    """Convert a raw scraped scores CSV to long format with Team and Year columns.
    Excludes AA — it is derived as VT+UB+BB+FX in _add_aa_factor."""
    df = pd.read_csv(filepath)
    df_long = df.melt(
        id_vars=['GymnastID', 'Team', 'HomeAway'],
        value_vars=['VT', 'UB', 'BB', 'FX'],
        var_name='Event',
        value_name='Score'
    )
    df_long['Year'] = year
    return df_long.dropna(subset=['Score'])


def _compute_homeaway_factors(df, groupby_cols):
    """
    Given a long-format scores dataframe, compute weighted home/away factors.
    groupby_cols: columns to group by for the output factor (e.g. ['Event'] or ['Team', 'Event'])
    Returns a DataFrame with groupby_cols + ['homeaway_factor'].
    """
    # Per-gymnast home/away averages within each group
    gymnast_stats = (
        df.groupby(groupby_cols + ['GymnastID', 'HomeAway'])['Score']
        .agg(avg='mean', count='count')
        .reset_index()
    )

    gymnast_pivot = gymnast_stats.pivot_table(
        index=groupby_cols + ['GymnastID'],
        columns='HomeAway',
        values=['avg', 'count']
    )
    gymnast_pivot.columns = [f"{v}_{k}" for v, k in gymnast_pivot.columns]
    gymnast_pivot = gymnast_pivot.reset_index()

    # Only keep gymnasts with both home and away scores
    gymnast_pivot = gymnast_pivot.dropna(subset=['avg_H', 'avg_A'])
    gymnast_pivot['diff'] = gymnast_pivot['avg_H'] - gymnast_pivot['avg_A']
    # Harmonic mean weight — penalises imbalance less aggressively than min()
    gymnast_pivot['weight'] = (
        2 * gymnast_pivot['count_H'] * gymnast_pivot['count_A']
        / (gymnast_pivot['count_H'] + gymnast_pivot['count_A'])
    )

    # Weighted factor + effective sample size + within-group variance of diffs
    def group_stats(g):
        total_weight = g['weight'].sum()
        if total_weight == 0:
            return None
        factor = (g['diff'] * g['weight']).sum() / total_weight
        # Weighted variance of gymnast diffs (estimation noise)
        var = (g['weight'] * (g['diff'] - factor) ** 2).sum() / total_weight
        return pd.Series({'homeaway_factor': factor, 'total_weight': total_weight, 'within_var': var})

    factors = (
        gymnast_pivot.groupby(groupby_cols)
        .apply(group_stats, include_groups=False)
        .reset_index()
        .dropna(subset=['homeaway_factor'])
    )
    factors['homeaway_factor'] = factors['homeaway_factor'].clip(lower=0)
    return factors


def _shrink_towards_league(team_factors, league_factors, groupby_cols):
    """
    Empirical Bayes shrinkage of per-team factors towards the league mean.

    For each team+event:
        shrunk = league_mean + (1 - B) * (team_factor - league_mean)
        B = sigma2_i / (sigma2_i + tau2)

    where:
        sigma2_i = within-team estimation variance = within_var / total_weight
        tau2     = between-team variance (estimated from data, floored at 0)
    """
    other_cols = [c for c in groupby_cols if c != 'Event']
    league_dict = dict(zip(league_factors['Event'], league_factors['homeaway_factor']))

    result_rows = []
    for event, group in team_factors.groupby('Event'):
        league_mean = league_dict.get(event, 0)

        # Per-team estimation variance
        group = group.copy()
        group['sigma2'] = group['within_var'] / group['total_weight']

        # Between-team variance: observed variance of team factors minus mean estimation variance
        observed_var = group['homeaway_factor'].var(ddof=1) if len(group) > 1 else 0
        tau2 = max(0, observed_var - group['sigma2'].mean())

        # Shrinkage: B=1 → full shrinkage to league mean; B=0 → keep team estimate
        if tau2 == 0:
            group['homeaway_factor_shrunk'] = league_mean
        else:
            group['B'] = group['sigma2'] / (group['sigma2'] + tau2)
            group['homeaway_factor_shrunk'] = (
                league_mean + (1 - group['B']) * (group['homeaway_factor'] - league_mean)
            )

        group['homeaway_factor_shrunk'] = group['homeaway_factor_shrunk'].clip(lower=0)
        result_rows.append(group)

    shrunk = pd.concat(result_rows, ignore_index=True)
    keep_cols = groupby_cols + ['homeaway_factor', 'homeaway_factor_shrunk', 'total_weight']
    return shrunk[keep_cols]


def _add_aa_factor(factors, groupby_cols):
    """Add AA row(s) as the sum of VT+UB+BB+FX factors (for all numeric columns)."""
    event_factors = factors[factors['Event'].isin(['VT', 'UB', 'BB', 'FX'])]
    other_cols = [c for c in groupby_cols if c != 'Event']
    numeric_cols = [c for c in factors.columns if c not in groupby_cols]
    if other_cols:
        aa = event_factors.groupby(other_cols)[numeric_cols].sum().reset_index()
    else:
        aa = pd.DataFrame({c: [event_factors[c].sum()] for c in numeric_cols})
    aa['Event'] = 'AA'
    return pd.concat([factors, aa], ignore_index=True)


def calculate_homeaway_factor(
    year_csvs=None,
    output_team_csv="Files/team_homeaway_factor.csv",
    output_league_csv="Files/league_homeaway_factor.csv",
    output_yearly_csv="Files/yearly_homeaway_factor.csv"
):
    """
    Calculate home/away factors from 2022-2025 historical data.

    Args:
        year_csvs: dict of {year: filepath} for raw scraped CSVs.
                   Defaults to the standard historical + 2025 files.
        output_team_csv: per-team/event factor output path
        output_league_csv: league-wide/event factor output path (fallback)
        output_yearly_csv: per-year/event breakdown output path
    """
    if year_csvs is None:
        year_csvs = {
            2022: "Historical/scores_2022.csv",
            2023: "Historical/scores_2023.csv",
            2024: "Historical/scores_2024.csv",
            2025: "2025 files/road_to_nationals.csv",
        }

    # Load and combine all years
    frames = []
    for year, path in year_csvs.items():
        try:
            frames.append(_raw_to_long(path, year))
            print(f"Loaded {year}: {path}")
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping {year}")
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined dataset: {len(df)} rows across {df['Year'].nunique()} years")

    # League-wide/event factors (needed for shrinkage target)
    league_factors = _compute_homeaway_factors(df, ['Event'])
    league_factors = _add_aa_factor(league_factors, ['Event'])
    league_factors.to_csv(output_league_csv, index=False)
    print(f"Saved {output_league_csv}")
    print(f"  League-wide factors:\n{league_factors[['Event','homeaway_factor']].to_string(index=False)}")

    # Per-team/event factors with empirical Bayes shrinkage towards league mean
    team_factors_raw = _compute_homeaway_factors(df, ['Team', 'Event'])
    team_factors = _shrink_towards_league(team_factors_raw, league_factors, ['Team', 'Event'])
    team_factors = _add_aa_factor(team_factors, ['Team', 'Event'])
    # homeaway_factor_shrunk is the operative value used downstream
    team_factors.to_csv(output_team_csv, index=False)
    print(f"Saved {output_team_csv} ({len(team_factors)} rows)")

    # Per-year/event breakdown
    yearly_factors = _compute_homeaway_factors(df, ['Year', 'Event'])
    yearly_factors = _add_aa_factor(yearly_factors, ['Year', 'Event'])
    yearly_factors = yearly_factors.sort_values(['Event', 'Year'])
    yearly_factors.to_csv(output_yearly_csv, index=False)
    print(f"Saved {output_yearly_csv}")
    print(f"  Yearly breakdown:\n{yearly_factors[['Year','Event','homeaway_factor']].to_string(index=False)}")

    return team_factors, league_factors, yearly_factors


if __name__ == "__main__":
    clean_data()
# %%
