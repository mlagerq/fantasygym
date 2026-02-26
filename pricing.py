from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import re
import getpass
import unicodedata


def normalize_name(name):
    """Remove accents and special characters from a name."""
    # Replace special characters that don't decompose properly
    replacements = {
        'ø': 'o', 'Ø': 'O',
        'æ': 'ae', 'Æ': 'AE',
        'ß': 'ss',
        ''': "'", ''': "'",
        '–': '-', '—': '-',
    }
    for char, replacement in replacements.items():
        name = name.replace(char, replacement)
    # Normalize to decomposed form (separate base chars from combining marks)
    normalized = unicodedata.normalize('NFKD', name)
    # Encode to ASCII, ignoring non-ASCII (removes accents), then decode back
    return normalized.encode('ascii', 'ignore').decode('ascii')

# Prompt for credentials
email = input("Enter your Fantasizr email: ")
password = getpass.getpass("Enter your Fantasizr password: ")

# 🚀 Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 🌐 Open Fantasizr login page
login_url = "https://www.fantasizr.com/login"
driver.get(login_url)

# ⏳ Step 1: Click "Login with Email/Password" Button
try:
    login_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-provider-id='password']"))
    )
    login_button.click()
    print("Clicked the login button to open email/password form.")
except Exception as e:
    print(f"Error clicking login button: {e}")
    driver.quit()

# ⏳ Step 2: Enter Email
try:
    email_field = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".firebaseui-id-email"))
    )
    email_field.send_keys(email)

    # Click the "Next" button
    next_button = driver.find_element(By.CSS_SELECTOR, "button.firebaseui-id-submit")
    next_button.click()
    print("Entered email and clicked 'Next'.")
except Exception as e:
    print(f"Error entering email or clicking 'Next': {e}")
    driver.quit()

# ⏳ Step 3: Enter Password
try:
    password_field = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".firebaseui-id-password"))
    )
    password_field.send_keys(password)

    # Click the "Sign In" button
    sign_in_button = driver.find_element(By.CSS_SELECTOR, "button.firebaseui-id-submit")
    sign_in_button.click()
    print("Entered password and clicked 'Sign In'.")
except Exception as e:
    print(f"Error entering password or clicking 'Sign In': {e}")
    driver.quit()

# ⏳ Wait for login to complete
try:
    WebDriverWait(driver, 5).until(EC.url_changes(login_url))
    print(f"Login successful! Current URL: {driver.current_url}")
except Exception as e:
    print(f"Login may have failed: {e}")
    driver.quit()

# 🎯 Navigate to the lineup edit page
lineup_url = "https://www.fantasizr.com/team/5023966516740096/lineup/edit"
driver.get(lineup_url)

# ⏳ Wait for lineup page to load
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".player-profile"))
    )
    print("Lineup page loaded successfully!")
except Exception as e:
    print(f"Error loading lineup page: {e}")
    driver.quit()


# 🔍 Wait for and find the class filter dropdown
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "class_filter"))
)
class_filter = Select(driver.find_element(By.ID, "class_filter"))
all_options = class_filter.options
print(f"Found {len(all_options)} filter options")

player_data = []

for i in range(len(all_options)):
    # Re-find the dropdown each iteration (DOM may refresh)
    class_filter = Select(driver.find_element(By.ID, "class_filter"))
    option_text = class_filter.options[i].text
    print(f"Selecting filter option: {option_text}")
    class_filter.select_by_index(i)

    # Wait for the page to update after selection
    time.sleep(1)

    # Scrape player data for this filter option
    players = driver.find_elements(By.CSS_SELECTOR, "[x-html='player.player_name']")
    prices = driver.find_elements(By.CSS_SELECTOR, "[x-text*='player.player_price']")
    teams = driver.find_elements(By.CSS_SELECTOR, "[x-text='player.player_bio']")

    # 📊 Extract and clean data
    for player, price, team in zip(players, prices, teams):
        full_name = player.text.strip()

        # Skip empty entries
        if not full_name:
            continue

        # Extract event abbreviation using regex (e.g., "First Last (XX)")
        match = re.match(r"(.*)\s\((\w{2})\)$", full_name)
        if match:
            name_only = match.group(1)  # Extracted name without event
            event_abbr = match.group(2)  # Event abbreviation
        else:
            name_only = full_name  # If no event abbreviation found
            event_abbr = ""

        team_name = team.text.strip()

        # Parse team_name into year, team, and conference (e.g., "Senior - Stanford - ACC")
        team_parts = team_name.split(" - ")
        if len(team_parts) == 3:
            year = team_parts[0]
            team_only = team_parts[1]
            conference = team_parts[2]
        else:
            year = ""
            team_only = team_name
            conference = ""

        player_data.append({
            "Player Name": name_only,
            "Year": year,
            "Team": team_only,
            "Event": event_abbr,
            "Price": price.text.strip()
        })

# 📑 Convert to DataFrame
df = pd.DataFrame(player_data)

# 🔢 Convert 'Price' column to numeric by removing $ and ,
df['Price'] = df['Price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Remove rows with empty prices
empty_price_rows = df[df['Price'].isna()]
if not empty_price_rows.empty:
    print(f"Dropping {len(empty_price_rows)} rows with empty prices:")
    print(empty_price_rows[['Player Name', 'Team', 'Event']].to_string())
df = df.dropna(subset=['Price'])

# Clean player names (remove accents and special characters)
df['Player Name'] = df['Player Name'].apply(normalize_name)

# 🔽 Save to CSV (optional)
df.to_csv("Files/fantasizr_player_pricing.csv", index=False)

# 🖥️ Display the DataFrame
print(df.head())
print(df.tail())

# 🚪 Close the browser
driver.quit()
