#%%
import pandas as pd
import numpy as np


def clean_data(input_csv="Files/road_to_nationals.csv", week_1_start="2025-12-30"):
    """
    Clean and transform scraped Road to Nationals data.

    Args:
        input_csv: Path to raw scraped data
        week_1_start: Start date of week 1 (for calculating week numbers)

    Returns:
        DataFrame with cleaned, adjusted scores in long format

    Outputs:
        - Files/player_info.csv
        - Files/road_to_nationals_long.csv
        - Files/scores_long_adjusted.csv
    """
    # Load the scraped data
    df = pd.read_csv(input_csv)

    ## Create dataframe player_info
    player_info = df[['GymnastID', 'Name', 'Team']].drop_duplicates()
    player_info.to_csv("Files/player_info.csv", index=False)
    print(f"Saved Files/player_info.csv with {len(player_info)} gymnasts")

    ## Use date to infer week of competition
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Define the start of the first week
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
    df_melted = df.melt(id_vars=['GymnastID', 'HomeAway', 'Week', 'Date', 'MeetID'],
                         value_vars=['VT', 'UB', 'BB', 'FX', 'AA'],
                         var_name='Event',
                         value_name='Score')

    # Save the reformatted DataFrame
    df_melted.to_csv("Files/road_to_nationals_long.csv", index=False)
    print(f"Saved Files/road_to_nationals_long.csv with {len(df_melted)} rows")

    # Load the league homeaway factor from saved 2025 calculation
    league_homeaway_factor = pd.read_csv("Files/league_homeaway_factor_2025.csv")

    # Apply homeaway adjustment to scores
    df_adj = df_melted.dropna().copy()
    df_adj = df_adj.merge(
        league_homeaway_factor[['Event', 'homeaway_factor']],
        on='Event',
        how='left'
    )

    # Subtract homeaway factor from home meet scores
    df_adj['score_adj'] = np.where(
        df_adj['HomeAway'] == 'H',
        df_adj['Score'] - df_adj['homeaway_factor'],
        df_adj['Score']
    )

    df_adj.to_csv("Files/scores_long_adjusted.csv", index=False)
    print(f"Saved Files/scores_long_adjusted.csv with {len(df_adj)} rows")

    return df_adj


# ==============================================================================
# ONE-TIME SETUP FUNCTIONS (run once to generate league_homeaway_factor_2025.csv)
# ==============================================================================

def calculate_homeaway_factor(input_csv="2025 files/road_to_nationals_long.csv", output_csv="Files/league_homeaway_factor_2025.csv"):
    """
    Calculate league-wide home/away factor from historical data.
    Only needs to be run once per season with prior year's data.
    """
    df_2025 = pd.read_csv(input_csv)
    df_2025 = df_2025.dropna()

    # Handle weeks with more than one meet for the same gymnast
    df_2025_agg = df_2025.groupby(['GymnastID','Week','Event']).agg(
        Score=pd.NamedAgg(column='Score', aggfunc='mean'),
        Home_Percent=pd.NamedAgg(column='HomeAway', aggfunc=(lambda x: sum(x == 'H')/sum(x == x))),
        Week_Weight=pd.NamedAgg(column='Score', aggfunc='count')
    )

    # Pivot to create week columns for scores
    df_2025_pivot = df_2025_agg.pivot_table(
        index=['GymnastID', 'Event'],
        columns='Week',
        values=['Score','Home_Percent','Week_Weight'],
        aggfunc='first'
    )
    df_2025_pivot.columns = [col[0] + "_" + str(col[1]) for col in df_2025_pivot.columns.values]

    # Identify score and homeaway columns
    score_cols_2025 = [col for col in df_2025_pivot.columns if 'Score' in col]
    homeaway_cols_2025 = [col for col in df_2025_pivot.columns if 'Home' in col]

    # Calculate average home and away scores per gymnast per event (row-wise)
    df_2025_pivot['home_avg'] = df_2025_pivot.apply(
        lambda row: row[[s for s, h in zip(score_cols_2025, homeaway_cols_2025) if row[h] == 1]].mean(),
        axis=1
    )
    df_2025_pivot['away_avg'] = df_2025_pivot.apply(
        lambda row: row[[s for s, h in zip(score_cols_2025, homeaway_cols_2025) if row[h] == 0]].mean(),
        axis=1
    )

    # Count home and away competitions for weighted average
    df_2025_pivot['home_count'] = df_2025_pivot.apply(
        lambda row: sum((row[h] == 1) and (pd.notna(row[s])) for s, h in zip(score_cols_2025, homeaway_cols_2025)),
        axis=1
    )
    df_2025_pivot['away_count'] = df_2025_pivot.apply(
        lambda row: sum((row[h] == 0) and (pd.notna(row[s])) for s, h in zip(score_cols_2025, homeaway_cols_2025)),
        axis=1
    )

    # Calculate difference and weighting factor
    df_2025_pivot['home_away_diff'] = df_2025_pivot['home_avg'] - df_2025_pivot['away_avg']
    df_2025_pivot['weight'] = df_2025_pivot[['home_count', 'away_count']].min(axis=1)

    # Group by event to calculate weighted league-wide average difference
    league_homeaway_factor = (
        df_2025_pivot.groupby('Event')
        .apply(lambda x: (
            (x['home_away_diff'] * x['weight']).sum() / x['weight'].sum()
            if x['weight'].sum() > 0 else None
        ), include_groups=False)
        .reset_index(name='homeaway_factor')
        .dropna(subset=['homeaway_factor'])
    )

    # Set negative factors to zero
    league_homeaway_factor['homeaway_factor'] = league_homeaway_factor['homeaway_factor'].clip(lower=0)

    # Add AA factor as sum of VT, UB, BB, FX factors
    aa_factor = league_homeaway_factor[league_homeaway_factor['Event'].isin(['VT', 'UB', 'BB', 'FX'])]['homeaway_factor'].sum()
    league_homeaway_factor = pd.concat([
        league_homeaway_factor,
        pd.DataFrame({'Event': ['AA'], 'homeaway_factor': [aa_factor]})
    ], ignore_index=True)

    league_homeaway_factor.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

    return league_homeaway_factor


if __name__ == "__main__":
    clean_data()
# %%
