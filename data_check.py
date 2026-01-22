import pandas as pd

# Load the CSV file
df = pd.read_csv("fantasizr_player_pricing.csv")

# Group by 'Player Name' and count the occurrences
player_counts = df.groupby('Player Name').size()

# Filter players with more than 4 entries
players_with_excess_entries = player_counts[player_counts > 4]

# Display players with more than 4 rows
if not players_with_excess_entries.empty:
    print("Players with more than 4 entries:")
    print(players_with_excess_entries)
else:
    print("All players have 4 or fewer entries.")

