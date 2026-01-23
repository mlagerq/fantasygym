"""
Fantasy Gymnastics Weekly Pipeline

Run this script each week to:
1. Scrape latest scores from Road to Nationals
2. Clean and transform the data
3. Run score and likelihood predictions
4. Optimize team selection
"""

from roadtonationals_scores import scrape_scores
from data_cleaning import clean_data
from predict import run_predictions
from team_optimizer import optimize_team


def run_weekly_pipeline(bye_teams=None, double_header_teams=None):
    """
    Run the complete weekly fantasy gymnastics pipeline.

    Args:
        bye_teams: List of team names with byes (gymnasts excluded from selection)
        double_header_teams: List of team names with double headers
    """
    print("=" * 60)
    print("STEP 1: Scraping scores from Road to Nationals")
    print("=" * 60)
    scrape_scores()

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning and transforming data")
    print("=" * 60)
    clean_data()

    print("\n" + "=" * 60)
    print("STEP 3: Running predictions")
    print("=" * 60)
    run_predictions()

    print("\n" + "=" * 60)
    print("STEP 4: Optimizing team selection")
    print("=" * 60)
    optimize_team(bye_teams=bye_teams, double_header_teams=double_header_teams)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("Check Files/lineup.csv for your optimized team")
    print("=" * 60)


def get_valid_teams():
    """Load valid team names from player_info.csv if it exists."""
    import os
    if os.path.exists("Files/player_info.csv"):
        import pandas as pd
        df = pd.read_csv("Files/player_info.csv")
        return set(df["Team"].unique())
    return None


def prompt_for_team_list(prompt_text, valid_teams):
    """Prompt user for a list of teams with validation."""
    while True:
        print(prompt_text)
        user_input = input("> ").strip()

        if not user_input:
            return []

        teams = [t.strip() for t in user_input.split(",") if t.strip()]

        if valid_teams is None:
            return teams

        # Validate teams
        invalid_teams = [t for t in teams if t not in valid_teams]
        if invalid_teams:
            print(f"\nInvalid team(s): {invalid_teams}")
            print(f"Valid teams: {sorted(valid_teams)}")
            print("Please try again.\n")
            continue

        return teams


def prompt_for_teams():
    """Prompt user to enter bye teams and double header teams."""
    valid_teams = get_valid_teams()

    if valid_teams:
        print(f"Found {len(valid_teams)} teams in player_info.csv\n")
    else:
        print("Note: player_info.csv not found, skipping team validation\n")

    bye_teams = prompt_for_team_list(
        "Enter teams with BYES this week (comma-separated, or press Enter for none):",
        valid_teams
    )

    print()
    double_header_teams = prompt_for_team_list(
        "Enter teams with DOUBLE HEADERS this week (comma-separated, or press Enter for none):",
        valid_teams
    )

    return bye_teams, double_header_teams


if __name__ == "__main__":
    bye_teams, double_header_teams = prompt_for_teams()
    print()
    run_weekly_pipeline(bye_teams=bye_teams, double_header_teams=double_header_teams)
