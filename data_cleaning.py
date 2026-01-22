#%%
import pandas as pd
import numpy as np

# Load the two CSVs into DataFrames
df = pd.read_csv("road_to_nationals.csv")
#df_fantasizr = pd.read_csv("fantasizr_player_pricing.csv")

## Create dataframe player_info
player_info = df[['GymnastID', 'Name', 'Team']].drop_duplicates()
player_info.to_csv("player_info.csv", index=False)

## Use date to infer week of competition
# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Define the start of the first week
week_1_start = pd.to_datetime("2025-12-30")  # Adjust the year if necessary

# Calculate the week number
df['Week'] = ((df['Date'] - week_1_start).dt.days // 7) + 1

# Remove invalid dates (if any)
df = df.dropna(subset=['Week'])

# Convert 'Week' to integer type
df['Week'] = df['Week'].astype(int)

#%%
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

#%%
# Transform scores df to have events on separate rows

# Melt the DataFrame to make one row per event per week per meet
df_melted = df.melt(id_vars=['GymnastID', 'HomeAway', 'Week', 'Date', 'MeetID'], 
                     value_vars=['VT', 'UB', 'BB', 'FX', 'AA'], 
                     var_name='Event', 
                     value_name='Score')

# Save the reformatted DataFrame
df_melted.to_csv("road_to_nationals_long.csv", index=False)

#%%
# SKIP THIS SECTION
df_melted = pd.read_csv("road_to_nationals_long.csv")

# Pivot to have one column per week's score
pivot_df = df_melted.pivot_table(
    index=['GymnastID', 'Event'], 
    columns='Week', 
    values='Score', 
    aggfunc='mean'
).reset_index()

# Rename week columns to weekX_score
pivot_df.columns = ['GymnastID', 'Event'] + [f'week{int(col)}_score' for col in pivot_df.columns[2:]]


# Calculate average, high, and low scores
pivot_df['average_score'] = pivot_df.loc[:, pivot_df.columns.str.contains('week')].mean(axis=1, skipna=True)
pivot_df['high_score'] = pivot_df.loc[:, pivot_df.columns.str.contains('week')].max(axis=1, skipna=True)
pivot_df['low_score'] = pivot_df.loc[:, pivot_df.columns.str.contains('week')].min(axis=1, skipna=True)


# Drop rows with no scores (all week columns are NaN)
score_columns = [col for col in pivot_df.columns if 'week' in col]
pivot_df = pivot_df.dropna(subset=score_columns, how='all')

# Merge with player_info using GymnastID
player_info = pd.read_csv("player_info.csv")
combined_df = pd.merge(player_info, pivot_df, on='GymnastID', how='inner')

# Load the combined gymnast summary and fantasizr datasets
fantasizr = pd.read_csv("fantasizr_player_pricing.csv")

# Ensure consistent name formatting (strip whitespace, lowercase)
combined_df['Name'] = combined_df['Name'].str.strip().str.lower()
fantasizr['Player Name'] = fantasizr['Player Name'].str.strip().str.lower()

# Merge the datasets on Name and Event (inner join to keep only matches)
final_df = pd.merge(
    combined_df, 
    fantasizr, 
    left_on=['Name', 'Event'], 
    right_on=['Player Name', 'Event'], 
    how='inner'
)

# Drop redundant columns (e.g., Player Name if duplicated)
final_df = final_df.drop(columns=['Player Name'])

# Save the final joined DataFrame
final_df.to_csv("gymnast_event_summary.csv", index=False)



#%%
# SKIP THIS SECTION
### Reformat data for model to have weekly scores and weekly home/away value

# Load 2025 data to calculate homeaway factor
df_2025 = pd.read_csv("2025 files/road_to_nationals_long.csv")
df_2025 = df_2025.dropna()

# Handle weeks with more than one meet for the same gymnast (2025 data for factor calculation)
df_2025_agg = df_2025.groupby(['GymnastID','Week','Event']).agg(
    Score=pd.NamedAgg(column='Score', aggfunc='mean'),
    Home_Percent=pd.NamedAgg(column='HomeAway', aggfunc=(lambda x: sum(x == 'H')/sum(x == x))),
    Week_Weight=pd.NamedAgg(column='Score', aggfunc='count')
)

# Pivot to create week columns for scores (2025 data)
df_2025_pivot = df_2025_agg.pivot_table(
    index=['GymnastID', 'Event'],
    columns='Week',
    values=['Score','Home_Percent','Week_Weight'],
    aggfunc='first'
)
df_2025_pivot.columns = [col[0] + "_" + str(col[1]) for col in df_2025_pivot.columns.values]

### Calculate HomeAway_Factor from 2025 data
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

# Count home and away competitions for weighted average (only if score is not null)
df_2025_pivot['home_count'] = df_2025_pivot.apply(
    lambda row: sum((row[h] == 1) and (pd.notna(row[s])) for s, h in zip(score_cols_2025, homeaway_cols_2025)),
    axis=1
)

df_2025_pivot['away_count'] = df_2025_pivot.apply(
    lambda row: sum((row[h] == 0) and (pd.notna(row[s])) for s, h in zip(score_cols_2025, homeaway_cols_2025)),
    axis=1
)

# Calculate difference and weighting factor (use minimum of home and away counts)
df_2025_pivot['home_away_diff'] = df_2025_pivot['home_avg'] - df_2025_pivot['away_avg']
df_2025_pivot['weight'] = df_2025_pivot[['home_count', 'away_count']].min(axis=1)

# Group by event to calculate weighted league-wide average difference (from 2025 data)
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

# Save the league homeaway factor 
league_homeaway_factor.to_csv("league_homeaway_factor_2025.csv",index=False)

#%%
# Now process 2026 data for weekly_scores_adjusted.csv (this is the input for mixed effects model I think?)

# Load the league homeaway factor from saved 2025 calculation
league_homeaway_factor = pd.read_csv("league_homeaway_factor_2025.csv")

# Load the melted DataFrame (2026 data)
df_melted = pd.read_csv("road_to_nationals_long.csv")
df_melted = df_melted.dropna()

# Handle weeks with more than one meet for the same gymnast (2025 data for factor calculation)
df_melted_agg = df_melted.groupby(['GymnastID','Week','Event']).agg(
    Score=pd.NamedAgg(column='Score', aggfunc='mean'),
    Home_Percent=pd.NamedAgg(column='HomeAway', aggfunc=(lambda x: sum(x == 'H')/sum(x == x)))
)

# Pivot to create week columns for scores (2026 data)
df = df_melted_agg.pivot_table(
    index=['GymnastID', 'Event'],
    columns='Week',
    values=['Score','Home_Percent'],
    aggfunc='first'
)
df.columns = [col[0] + "_" + str(col[1]) for col in df.columns.values]

# Identify score and homeaway columns for 2026 data
score_cols = [col for col in df.columns if 'Score' in col]
homeaway_cols = [col for col in df.columns if 'Home' in col]

# Merge homeaway_factor into the main dataframe by Event
df = df.reset_index()  # Convert GymnastID index to column
df_merged = pd.merge(df, league_homeaway_factor, on='Event', how='left', validate='many_to_one')

# Subtract homeaway_factor from every home score
for score_col, home_col in zip(score_cols, homeaway_cols):
    df_merged[score_col] = df_merged.apply(
        lambda row: row[score_col] - (row['homeaway_factor'])*row[home_col],
        axis=1
    )

# Save the adjusted dataframe
df_merged.to_csv("weekly_scores_adjusted.csv", index=False)

#%%
# Save road_to_nationals_long with the adjusted scores (for predict.py)
# Load the DataFrame (2026 data)
df = pd.read_csv("road_to_nationals_long.csv")

## Add league homeaway factor based on event to each row in df
df = df.dropna()
df = df.merge(
    league_homeaway_factor[['Event', 'homeaway_factor']],
    on='Event',
    how='left'
)

## Subtract homeaway factor from home meet scores
df['score_adj'] = np.where(
    df['HomeAway'] == 'H',
    df['Score'] - df['homeaway_factor'],
    df['Score']
)

df.to_csv("scores_long_adjusted.csv", index=False)
# %%
# plan
# find league home/away split (could use intraplayer splits for more accuracy, weight by harmonic mean of home/away matches)
# adjust scores for home away
# combine multi meet weeks somehow - potentially add a weight column

# replace NAs with 0s?

# create DF with data to predict gymnast_id, week, event, score, prev_score_0, prev_score_1, prev_score_2, ..., compete_0, compete_1, compete_2, ..., percent_compete, other explanatory variables, matches_1, matches_2, ...
# test out different models:
# from chat gpt try 3 model types 
### mixed model (most basic probably)
##### uses partial pooling to group similar gymnasts and handles missing data nicely
### beysian (STAN)
##### also uses partial pooling, impose more prior knowledge of features 
### xgboost/GAM (tree)
##### generate parameters - do more feature engineering, can be tolerant of missingness but I'd have to do something to indicate it??
##### maybe the one john said to make a multi-week average for?
### Panel/Longitudinal Data Models
##### explicit time series
# after model, unadjust home/away & account for multiple meets
# how to predict at different points in the season? probably fine if I engineer a feature for past scores instead of weekly data
# ? predict for 2 meet gymnasts (mean of std of gymnasts -> biased high since more kept constant in same week meets, could just look at meets in the same week, std changes by skill level (higher worse))

