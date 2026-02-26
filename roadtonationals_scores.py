#%%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException  # Import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re

#%%
def scrape_scores(fantasizr_csv="Files/fantasizr_player_pricing.csv", output_csv="Files/road_to_nationals.csv"):
    """
    Scrape gymnast scores from Road to Nationals website.

    Args:
        fantasizr_csv: Path to Fantasizr pricing CSV (used to filter teams)
        output_csv: Path to save scraped scores

    Returns:
        DataFrame with scraped scores
    """
    # Set up Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run without UI
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Open the website to any team and gymnast (but not the first team)
        url = "https://roadtonationals.com/results/teams/gymnast/2026/3/32927"
        driver.get(url)

        # Wait for the team dropdown to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "rt-td"))
        )

        # Find the team dropdown menu
        team_dropdown = Select(driver.find_element(By.ID, "team_filter"))

        for option in team_dropdown.options:
            print(f"Value: '{option.get_attribute('value')}', Text: '{option.text.strip()}'")

        # Get all team IDs and Names as a dictionary {team_id: team_name}
        team_options = {
            option.get_attribute("value"): option.text.strip()
            for option in team_dropdown.options
            if option.get_attribute("value")  # Ensure it's not empty
        }

        print(team_options)

        # Load the Fantasizr CSV
        df_fantasizr = pd.read_csv(fantasizr_csv)

        # Extract unique team names and remove the conference information
        # Example: "Team Name (Conference)" -> "Team Name"
        df_fantasizr['Cleaned_Team'] = df_fantasizr['Team'].apply(lambda x: re.sub(r'\s*\(.*?\)', '', x).strip())

        # Get unique cleaned team names
        unique_teams = df_fantasizr['Cleaned_Team'].unique().tolist()

        # Display the list of unique teams
        print("Unique Teams in Fantasizr Dataset:")
        print(unique_teams)

        # Store all scraped data
        all_data = []

        for team_id, team_name in team_options.items():

            # Skip teams not in the Fantasizr dataset
            cleaned_team_name = re.sub(r'\s*\(.*?\)', '', team_name).strip()

            if cleaned_team_name not in unique_teams:
                print(f"Skipping {team_name} (not in Fantasizr dataset).")
                continue

            # Continue with the scraping process for valid teams...

            print(f"Scraping data for {team_name} (ID: {team_id})")

            # Re-fetch team dropdown to avoid stale element reference
            team_dropdown = Select(driver.find_element(By.ID, "team_filter"))

            # Capture the current state of the gymnast dropdown
            initial_gymnast_options = [option.text.strip() for option in Select(driver.find_element(By.ID, "gymnast_filter")).options]

            # Select team
            team_dropdown.select_by_value(team_id)

            # Wait until the gymnast dropdown updates with new options
            def gymnast_dropdown_updated(d):
                try:
                    current_options = [option.text.strip() for option in Select(d.find_element(By.ID, "gymnast_filter")).options if option.text.strip()]
                    return current_options != initial_gymnast_options
                except:
                    return False
            WebDriverWait(driver, 10).until(gymnast_dropdown_updated)

            # Find the updated gymnast dropdown menu
            dropdown = Select(driver.find_element(By.ID, "gymnast_filter"))

            # Get all gymnast IDs and Names as a dictionary {gymnast_id: gymnast_name}
            gymnast_options = {
                option.get_attribute("value"): option.text.strip()
                for option in dropdown.options
                if option.get_attribute("value")  # Ensure it's not empty
            }

            # Loop through all gymnast options
            for gymnast_id, gymnast_name in gymnast_options.items():
                print(f"Scraping data for {gymnast_name} (ID: {gymnast_id})")

                # Select gymnast
                dropdown.select_by_value(gymnast_id)

                # Wait for data to load
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".rt-tr.-odd, .rt-tr.-even, .rt-noData"))
                )

                # Extract all table rows
                rows = driver.find_elements(By.CSS_SELECTOR, ".rt-tr.-odd, .rt-tr.-even")

                # Scrape row data
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "div")
                    row_data = [col.text.strip() for col in cols]
                    row_data.append(team_name)  # Add team name
                    row_data.append(gymnast_id)  # Add Gymnast ID
                    row_data.append(gymnast_name)  # Add Gymnast Name
                    all_data.append(row_data)

        # Convert to Pandas DataFrame
        df = pd.DataFrame(all_data)

        # Drop the 3rd column (index 2)
        df = df.drop(df.columns[2], axis=1)

        # Add column names (modify as needed)
        df.columns = ["Date", "Opponent", "VT", "UB", "BB", "FX", "AA", "HomeAway", "Team", "GymnastID", "Name"]

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} rows to {output_csv}")

        return df

    finally:
        # Close the browser
        driver.quit()

#%%
def scrape_schedule(week, fantasizr_csv="Files/fantasizr_player_pricing.csv"):
    """
    Scrape meet schedule from Road to Nationals website.

    Args:
        week: Week number to scrape schedule for
        fantasizr_csv: Path to Fantasizr pricing CSV (used to filter teams)

    Returns:
        List of matchup strings (e.g., "Team A @ Team B")
    """
    # Set up Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run without UI
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    all_data = []
    try:
        # Open the website with dynamic week number
        url = f"https://roadtonationals.com/results/schedule/{week}"
        driver.get(url)
        print(f"Scraping schedule for week {week}: {url}")

        # Wait for the page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "team_filter"))
        )

        # Scrape matchup data
        rows = driver.find_elements(By.CSS_SELECTOR, ".schedule-link")
        for row in rows:
            row_data = row.text.strip()
            if row_data:
                all_data.append(row_data)
                print(f"  {row_data}")

        return all_data

    finally:
        # Close the browser
        driver.quit()


def parse_schedule(matchups, valid_teams):
    """
    Parse schedule matchups to determine home teams, double headers, and byes.

    Args:
        matchups: List of matchup strings (e.g., "Team A @ Team B")
        valid_teams: Set of valid team names from player_info.csv

    Returns:
        dict with keys: 'home_teams', 'double_header_teams', 'bye_teams', 'home_counts'
              home_counts is a dict of {team: number of home meets this week}
    """
    from collections import defaultdict

    # Normalize valid_teams to remove duplicate aliases
    valid_teams = normalize_valid_teams(valid_teams)

    team_appearances = defaultdict(int)  # How many times each team appears
    home_counts = defaultdict(int)  # How many home meets each team has

    for matchup in matchups:
        # Parse "Team A @ Team B" format - Team B is home
        # Also handles "Team A, Team B, Team C @ Team D" (multiple away teams)
        if " @ " in matchup:
            parts = matchup.split(" @ ")
            if len(parts) == 2:
                away_part = parts[0].strip()
                home_team = parts[1].strip()

                # Handle multiple away teams separated by commas
                away_teams = [t.strip() for t in away_part.split(",")]

                # Match home team to valid teams
                home_matched = match_team_name(home_team, valid_teams)
                if home_matched:
                    team_appearances[home_matched] += 1
                    home_counts[home_matched] += 1

                # Match each away team
                for away_team in away_teams:
                    away_matched = match_team_name(away_team, valid_teams)
                    if away_matched:
                        team_appearances[away_matched] += 1
        else:
            # Handle neutral site meets (e.g., "Alabama, Arizona, LSU, North Carolina (Purple & Gold)")
            # Split by comma and try to match each team
            teams = [t.strip() for t in matchup.split(",")]
            for team in teams:
                # Remove event name in parentheses (e.g., "(Purple & Gold)")
                team_clean = re.sub(r'\s*\([^)]*\)\s*$', '', team).strip()
                team_matched = match_team_name(team_clean, valid_teams)
                if team_matched:
                    team_appearances[team_matched] += 1

    # Determine categories
    home_teams = [team for team, count in home_counts.items() if count > 0]
    double_header_teams = [team for team, count in team_appearances.items() if count >= 2]
    bye_teams = [team for team in valid_teams if team not in team_appearances]

    print(f"\nSchedule Analysis:")
    print(f"  Home teams: {home_teams}")
    print(f"  Double header teams: {double_header_teams}")
    print(f"  Bye teams: {bye_teams}")

    return {
        'home_teams': home_teams,
        'double_header_teams': double_header_teams,
        'bye_teams': bye_teams,
        'home_counts': dict(home_counts)
    }


# Mapping from Road to Nationals schedule names to Fantasizr team names
SCHEDULE_TO_FANTASIZR = {
    'North Carolina State': 'NC State',
    'Pittsburgh': 'Pitt',
    'Ithaca College': 'Ithaca',
}

# Teams that are duplicated in Fantasizr data under different names
# Maps alternate name -> canonical name
TEAM_ALIASES = {
    'North Carolina State': 'NC State',
}


def normalize_valid_teams(valid_teams):
    """
    Remove duplicate team entries that are aliases of each other.
    Returns a set with only canonical team names.
    """
    return {TEAM_ALIASES.get(team, team) for team in valid_teams}


def match_team_name(name, valid_teams):
    """
    Match a team name from schedule to valid team names.
    Uses hardcoded mapping for known differences, then falls back to exact match.
    """
    import re
    name = name.strip()

    # Remove trailing scores (e.g., "Cornell 191.6500" -> "Cornell")
    name = re.sub(r'\s+\d+\.\d+$', '', name)

    # Check hardcoded mapping first
    if name in SCHEDULE_TO_FANTASIZR:
        mapped = SCHEDULE_TO_FANTASIZR[name]
        if mapped in valid_teams:
            return mapped

    # Direct match
    if name in valid_teams:
        return name

    # Case-insensitive match
    name_lower = name.lower()
    for team in valid_teams:
        if team.lower() == name_lower:
            return team

    # Not a team we track in Fantasizr
    return None

#%%
if __name__ == "__main__":
    scrape_scores()