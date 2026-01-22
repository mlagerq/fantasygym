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

# Filter out gymnasts with low likelihood to compete after we select base team
df = df[df['pred_prob'] >= 0.7]

#%%
# Create a base team by taking the best of the cheapest
# Sort by price (ascending) then pred_score (descending) to get cheapest first, best among ties
def select_cheapest_best(event_df):
    event_df = event_df.sort_values(by=['Price', 'pred_score'], ascending=[True, False])
    return event_df.head(5)

base_team = df[['GymnastID','Name','Team','Event','Price','pred_score','pred_prob']].groupby('Event').apply(select_cheapest_best).reset_index(drop=True)
team_score = base_team['pred_score'].sum()
team_cost = base_team['Price'].sum()

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
    worst_gymnast = event_lineup_df.loc[event_lineup_df['pred_score'].idxmin()]
    worst_gymnast_name = worst_gymnast['Name']
    worst_weight = worst_gymnast['pred_score']
    worst_cost = worst_gymnast['Price']

    # remove any gymnasts from full_df of the same or lower price to the worst team member
    event_df = event_df[(event_df['Price'] > worst_cost) & (event_df['pred_score'] > worst_weight)]
    # calculate value of all gymnasts compared to the current worst team member
        ### value = 1000 * (weight - low_weight) / (price - low_price)
    event_df['Value'] = 1000 * (event_df['pred_score'] - worst_weight) / (event_df['Price'] - worst_cost)
    event_df = event_df.sort_values(by='Value', ascending=False).reset_index()
    if event_df.empty:
        print('Event_df for ' + event + ' is empty')
        return None
    else:
        best_replacement = {
            'Name': event_df.loc[0, 'Name'],
            'pred_score': event_df.loc[0, 'pred_score'],
            'Price': event_df.loc[0, 'Price'],
            'Value': event_df.loc[0, 'Value']
        }
    best_replacement_for_event = (event, worst_gymnast_name, best_replacement['Name'], best_replacement['pred_score'] - worst_weight, best_replacement['Price'] - worst_cost, best_replacement['Value'])
    return best_replacement_for_event

# Function to replace gymnasts to maximize score while minimizing cost
def replace_gymnasts_to_target_cost(base_team_df, full_df, target_cost=92500):
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
        updated_df.sort_values(by=['Event','pred_score'],ascending=[True,False]).reset_index(drop=True)
        total_cost = calculate_total_cost(updated_df)
        print(f"Replacement made: {replacement_name} for {worst_name} on {event}, New cost: {total_cost}")
    
    return updated_df, total_cost

#%%
# Apply the function to replace gymnasts iteratively until cost <= budget
final_df, final_cost = replace_gymnasts_to_target_cost(base_team, df)

#%%
# Add the best gymnast priced at 1500 for each event as the 6th roster spot
def add_best_cheap_gymnast(final_df, full_df, price=1500):
    updated_df = final_df.copy()

    for event in updated_df['Event'].unique():
        # Get gymnasts already on the team for this event
        gymnasts_in_lineup = updated_df[updated_df['Event'] == event]['Name'].tolist()

        # Find all gymnasts priced at 1500 for this event who aren't on the team
        cheap_gymnasts = full_df[
            (full_df['Event'] == event) &
            (full_df['Price'] == price) &
            (~full_df['Name'].isin(gymnasts_in_lineup))
        ]

        if not cheap_gymnasts.empty:
            # Get the best one by pred_score
            best_cheap = cheap_gymnasts.loc[cheap_gymnasts['pred_score'].idxmax()]
            # Add to team
            updated_df = pd.concat([updated_df, pd.DataFrame([best_cheap])], ignore_index=True)
            print(f"Added 6th gymnast for {event}: {best_cheap['Name']} (score: {best_cheap['pred_score']:.3f})")
        else:
            print(f"No available gymnast priced at {price} for {event}")

    return updated_df

final_df = add_best_cheap_gymnast(final_df, df)
final_cost = calculate_total_cost(final_df)

# TODO: deal with extra budget - find the next best score to replace without worrying about value
# TODO: add param for teams to skip (byes)
# TODO: add param for teams that play twice - need to estimate variance then mean + 0.8SD

# Output the final dataframe and cost
final_df = final_df.sort_values(by=['Event','pred_score'], ascending=[True,False]).reset_index(drop=True)
print(final_df)
print(f"Total Cost: {final_cost}")
print(f"Total Score: {final_df.groupby('Event').apply(lambda x: x.nlargest(5, 'pred_score')).reset_index(drop=True)['pred_score'].sum()}")

# %%
final_df.to_csv('lineup.csv',index=False)
# %%
