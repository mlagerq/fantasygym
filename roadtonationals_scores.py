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


def scrape_scores(fantasizr_csv="fantasizr_player_pricing.csv", output_csv="road_to_nationals.csv"):
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


if __name__ == "__main__":
    scrape_scores()
# %%
