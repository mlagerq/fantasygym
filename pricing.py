from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import re
import getpass

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


# 🔍 Scrape player data
players = driver.find_elements(By.CSS_SELECTOR, "[x-html='player.player_name']")
prices = driver.find_elements(By.CSS_SELECTOR, "[x-text*='player.player_price']")
teams = driver.find_elements(By.CSS_SELECTOR, "[x-text='player.player_bio']")

# 📊 Extract data
# 📊 Extract and clean data
player_data = []
for player, price, team in zip(players, prices, teams):
    full_name = player.text.strip()
    
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
df['Price'] = df['Price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# 🔽 Save to CSV (optional)
df.to_csv("fantasizr_player_pricing_2026.csv", index=False)

# 🖥️ Display the DataFrame
print(df.head())
print(df.tail())

# 🚪 Close the browser
driver.quit()
