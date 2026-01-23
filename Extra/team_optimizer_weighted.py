# make the best team regardless of price
# substitute gymnasts of the least score loss until you get to < 90k
#%%
import pandas as pd

df = pd.read_csv("team_opt_input_linear.csv")
df.rename(columns={"Player Name": "Name"},inplace=True)

df = df.groupby(['GymnastID', 'Event'], as_index=False).agg({
    'pred_score': 'mean',  # Use 'mean', or 'max' depending on your preference
    'pred_prob': 'mean',
    'Price': 'first', 
    'Name': 'first',  
    'Team': 'first'
})

# Weight predicted score by likelihood to compete
df['Weight'] = df['pred_score']*df['pred_prob']

# Create a base team by taking the best of the cheapest
base_team = df[df['Price']==1000]
base_team = base_team[['GymnastID','Name','Team','Event','Weight','Price','pred_score','pred_prob']].groupby('Event').apply(lambda x: x.nlargest(5, 'Weight')).reset_index(drop=True)
team_score = base_team['Weight'].sum() # this is really the team weighted score
team_cost = base_team['Price'].sum() # should be 20000

# Function to calculate the total cost of the lineup
def calculate_total_cost(df):
    return df['Price'].sum()

# Function to find the best value replacement for a gymnast in a specific event
# Inputs: current team, event (will loop through all), all gymnasts
def find_best_replacement_for_event(updated_df, event, full_df):
    # Current lineup for this event
    event_lineup_df = updated_df[updated_df['Event'] == event]
    # List of gymnast names in the lineup
    gymnasts_in_lineup = event_lineup_df['Name'].tolist()
    # All gymnasts on that event
    event_df = full_df[full_df['Event'] == event]
    # Remove gymnasts who are already in the lineup - now just eligible replacements
    event_df = event_df[~event_df['Name'].isin(gymnasts_in_lineup)]
    
    # identify the worst team member (lowest weight)
    worst_gymnast = event_lineup_df.loc[event_lineup_df['Weight'].idxmin()]
    worst_gymnast_name = worst_gymnast['Name']
    worst_weight = worst_gymnast['Weight']
    worst_cost = worst_gymnast['Price']

    # remove any gymnasts from full_df of the same or lower price to the worst team member
    event_df = event_df[(event_df['Price'] > worst_cost) & (event_df['Weight'] > worst_weight)]
    # calculate value of all gymnasts compared to the current worst team member
        ### value = 1000 * (weight - low_weight) / (price - low_price)
    event_df['Value'] = 1000 * (event_df['Weight'] - worst_weight) / (event_df['Price'] - worst_cost)
    event_df = event_df.sort_values(by='Value', ascending=False).reset_index()
    best_replacement = {
            'Name': event_df.loc[0, 'Name'],
            'Weight': event_df.loc[0, 'Weight'],
            'Price': event_df.loc[0, 'Price'],
            'Value': event_df.loc[0, 'Value']
        }
    best_replacement_for_event = (event, worst_gymnast_name, best_replacement['Name'], best_replacement['Weight'] - worst_weight, best_replacement['Price'] - worst_cost, best_replacement['Value'])
    return best_replacement_for_event

# Function to replace gymnasts to maximize score while minimizing cost
def replace_gymnasts_to_target_cost(base_team_df, full_df, target_cost=86000):
    updated_df = base_team_df.copy()
    total_cost = calculate_total_cost(updated_df)

    while total_cost < target_cost:
        best_replacements = []

        # Find best replacement for each event
        for event in updated_df['Event'].unique():
            # Returns event, worst gymnast name, best replacement name, weight diff, price diff, and replacement value
            best_replacement_for_event = find_best_replacement_for_event(updated_df, event, full_df)
            
            # If this is not empty, add the replacement for consideration
            if best_replacement_for_event:
                best_replacements.append(best_replacement_for_event)
        
        # only consider replacements that don't go over budget
        best_replacements = [r for r in best_replacements if r[4]<=(target_cost - total_cost)]
        if not best_replacements: break

        # pick the best value replacement across events
        best_replacements.sort(key=lambda x: x[5],reverse=True)  # Sort by value (make sure this is right)
        best_replacement_overall = best_replacements[0]
        replacement_name = best_replacement_overall[2]
        worst_name = best_replacement_overall[1]
        event = best_replacement_overall[0]

        # Replace the worst gymnast with the best replacement
        best_replacement = full_df[(full_df['Name'] == replacement_name) & (full_df['Event'] == event)]
        updated_df = pd.concat([updated_df,best_replacement],ignore_index=True)
        updated_df = updated_df[~((updated_df['Name'] == worst_name) & (updated_df['Event'] == event))]
        updated_df.sort_values(by=['Event','Weight'],ascending=[True,False]).reset_index(drop=True)
        total_cost = calculate_total_cost(updated_df)
        print(f"Replacement made: {replacement_name} for {worst_name} on {event}, New cost: {total_cost}")
    
    return updated_df, total_cost

# Apply the function to replace gymnasts iteratively until cost <= budget
final_df, final_cost = replace_gymnasts_to_target_cost(base_team, df)

# TODO: add in the best cheapest players as the 6th (that are not already on the team)
# TODO: try with a pred_prob cutoff instead of Weight calculation directly
# TODO: add param for teams to skip (byes)
# TODO: add param for teams that play twice - need to estimate variance then mean + 0.8SD

# Output the final dataframe and cost
final_df = final_df.sort_values(by=['Event','pred_score'], ascending=[True,False]).reset_index(drop=True)
print(final_df)
print(f"Total Cost: {final_cost}")
print(f"Total Score: {final_df.groupby('Event').apply(lambda x: x.nlargest(5, 'pred_score')).reset_index(drop=True)['Weight'].sum()}")

#%%













#%%
# OLD METHOD
import pandas as pd

df = pd.read_csv("team_opt_input.csv")

# Aggregate the rows by GymnastID and Event, taking the mean of the predicted scores (or any other aggregation method)
# I think this is because some gymnasts competed twice in a week?
df = df.groupby(['GymnastID', 'Event'], as_index=False).agg({
    'Predicted_Score': 'mean',  # Use 'mean', or 'max' depending on your preference
    'Price': 'first', 
    'Name': 'first',  
    'Team': 'first'
})
top_team = df[['Name','Event','Predicted_Score','Price']].groupby('Event').apply(lambda x: x.nlargest(5, 'Predicted_Score')).reset_index(drop=True)
team_score = top_team['Predicted_Score'].sum()
team_cost = top_team['Price'].sum()
    
# %% 
# Function to calculate the total cost of the lineup
def calculate_total_cost(df):
    return df['Price'].sum()

# Function to find the best replacement for a gymnast at a given price range in a specific event
# Inputs: current top team, event (will loop through all), all gymnasts
def find_best_replacement_for_event(updated_df, event, full_df):
    # Sort the full dataframe for the event by price (ascending), then predicted score (descending)
    gymnasts_in_lineup = updated_df[updated_df['Event'] == event]['Name'].tolist()
    event_df = full_df[full_df['Event'] == event].sort_values(by=['Price', 'Predicted_Score'], ascending=[True, False])
    event_df = event_df[~event_df['Name'].isin(gymnasts_in_lineup)]

    # Iterate through the fixed price points
    price_points = [6250, 5000, 4000, 3000, 2000, 1000]
    
    best_replacement_for_event = []
    
    for price in price_points:
        # Get the gymnasts currently selected in the lineup for this event at the given price
        gymnasts_at_price = updated_df[(updated_df['Event'] == event) & (updated_df['Price'] == price)]

        if gymnasts_at_price.empty:
            continue
        
        # Select the gymnast with the lowest score at this price point (worst gymnast)
        worst_gymnast = gymnasts_at_price.loc[gymnasts_at_price['Predicted_Score'].idxmin()]
        worst_gymnast_name = worst_gymnast['Name']
        worst_score = worst_gymnast['Predicted_Score']
        worst_cost = worst_gymnast['Price']
        
        # Group the event dataframe by price and get the best replacement from each price group
        event_df_grouped_by_price = event_df[event_df['Price'] < price].groupby('Price').first().reset_index()

        if event_df_grouped_by_price.empty:
            continue
        
        max_index = event_df_grouped_by_price['Predicted_Score'].idxmax()
        best_replacement = {
            'Name': event_df_grouped_by_price.loc[max_index, 'Name'],
            'Event': event,
            'Predicted_Score': event_df_grouped_by_price.loc[max_index, 'Predicted_Score'],
            'Price': event_df_grouped_by_price.loc[max_index, 'Price']
        }
        replacement_score = best_replacement['Predicted_Score']
        replacement_cost = best_replacement['Price']
        
        # Calculate the impact on score and cost
        score_diff = worst_score - replacement_score  # Smaller is better
        cost_diff = worst_cost - replacement_cost     # Larger is better
        value_diff = 1000 * score_diff / cost_diff # Points per $, Smaller is better
        # Track the best replacement for minimizing score loss and maximizing cost savings
        best_replacement_for_event.append((worst_gymnast_name, best_replacement, score_diff, cost_diff, value_diff))
        
    return best_replacement_for_event


# Function to replace gymnasts to maximize score while minimizing cost
def replace_gymnasts_to_target_cost(top_6_df, full_df, target_cost=86000):
    updated_df = top_6_df.copy()
    total_cost = calculate_total_cost(updated_df)
    
    while total_cost > target_cost:
        best_replacements = []

        # Find best replacement for each event
        for event in updated_df['Event'].unique():
            best_replacement_for_event = find_best_replacement_for_event(updated_df, event, full_df)
            
            # Sort replacements by the least score loss and maximize cost savings
            best_replacement_for_event.sort(key=lambda x: (x[4]))  # Sort by value ascending (1000 * score impact / cost savings (points per ruple))
            
            if best_replacement_for_event:
                best_replacements.append(best_replacement_for_event[0])
        
        best_replacements.sort(key=lambda x: (x[4]))  # Sort by value
        best_replacement_overall = best_replacements[0]

        # Replace the worst gymnast with the best replacement
        event = best_replacement_overall[1]['Event']
        name = best_replacement_overall[0]
        best_replacement = best_replacement_overall[1]
        updated_df.loc[(updated_df['Name'] == name) & (updated_df['Event'] == event), 'Name'] = best_replacement['Name']
        updated_df.loc[(updated_df['Name'] == best_replacement['Name']) & (updated_df['Event'] == event), 'Predicted_Score'] = best_replacement['Predicted_Score']
        updated_df.loc[(updated_df['Name'] == best_replacement['Name']) & (updated_df['Event'] == event), 'Price'] = best_replacement['Price']
        
        total_cost = calculate_total_cost(updated_df)
        
        # If cost is now within the target, exit
        if total_cost <= target_cost:
            cheapest_df = full_df[full_df['Price']==1000]
            alternates = cheapest_df[['Name','Event','Predicted_Score','Price']].groupby('Event').apply(lambda x: x.nlargest(1, 'Predicted_Score')).reset_index(drop=True)
            updated_df = pd.concat([updated_df,alternates]).sort_values('Event').reset_index(drop=True)
            total_cost = calculate_total_cost(updated_df)
            break
    
    return updated_df, total_cost


# Apply the function to replace gymnasts iteratively until cost <= 90,000
final_df, final_cost = replace_gymnasts_to_target_cost(top_team, df)

# Output the final dataframe and cost
print(final_df)
print(f"Total Cost: {final_cost}")
print(f"Total Score: {final_df.groupby('Event').apply(lambda x: x.nlargest(5, 'Predicted_Score')).reset_index(drop=True)['Predicted_Score'].sum()}")

# %%
winners = pd.merge(
    final_df[['Name','Event','Price','Predicted_Score']],
    df[['Name','Event','Team']],
    left_on=['Name','Event'],
    right_on=['Name','Event'],
    how='inner'
)
# %%
winners.to_csv("Week_11_Team.csv")
# %%
