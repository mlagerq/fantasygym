#%%
import pandas as pd

week9 = pd.read_csv("Week_9_Team.csv")
df = pd.read_csv("road_to_nationals_long.csv")
players = pd.read_csv("player_info.csv")
df9 = df[df['Week']==9]

#%%
merged_df = pd.merge(
    week9[['Name','Event','Price','Team','Predicted_Score']],
    players[['Name','GymnastID']],
    left_on = ['Name'],
    right_on = ['Name'],
    how = 'inner'
)

final_df = pd.merge(
    merged_df,
    df9[['GymnastID','Event','Score']],
    left_on = ['GymnastID','Event'],
    right_on = ['GymnastID','Event'],
    how = 'inner'
)

#%%
print(f"Predicted Score: {final_df['Predicted_Score'].sum()}")
print(f"Actual Score: {final_df['Score'].sum()}")

# %%
final_df.to_csv("Week_9_Results.csv",index=False)
# %%
