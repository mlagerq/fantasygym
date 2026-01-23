"""
Fantasy Gymnastics Weekly Pipeline

Run this script each week to:
1. Scrape latest scores from Road to Nationals
2. Clean and transform the data
3. Run score and likelihood predictions
4. Optimize team selection
"""

from roadtonationals_scores import scrape_scores, scrape_schedule, parse_schedule
from data_cleaning import clean_data
from predict import run_predictions
from team_optimizer import optimize_team
import pandas as pd
import os


def run_weekly_pipeline(bye_teams=None, double_header_teams=None, home_teams=None, home_counts=None):
    """
    Run the complete weekly fantasy gymnastics pipeline.

    Args:
        bye_teams: List of team names with byes (gymnasts excluded from selection)
        double_header_teams: List of team names with double headers
        home_teams: List of team names competing at home
        home_counts: Dict of {team: number of home meets this week}
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
    optimize_team(
        bye_teams=bye_teams,
        double_header_teams=double_header_teams,
        home_teams=home_teams,
        home_counts=home_counts
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("Check Files/lineup.csv for your optimized team")
    print("=" * 60)


def get_valid_teams():
    """Load valid team names from player_info.csv if it exists."""
    if os.path.exists("Files/player_info.csv"):
        df = pd.read_csv("Files/player_info.csv")
        return set(df["Team"].unique())
    return None


def get_current_week():
    """Get the current week number from scores data."""
    if os.path.exists("Files/scores_long_adjusted.csv"):
        df = pd.read_csv("Files/scores_long_adjusted.csv")
        return int(df["Week"].max())
    return None


def get_schedule_info(next_week, valid_teams):
    """
    Scrape and parse schedule to get team status for next week.

    Args:
        next_week: Week number to scrape schedule for
        valid_teams: Set of valid team names

    Returns:
        dict with home_teams, double_header_teams, bye_teams, home_counts
    """
    print(f"\nScraping schedule for week {next_week}...")
    matchups = scrape_schedule(next_week)

    if not matchups:
        print("No matchups found in schedule. Using manual input.")
        return None

    return parse_schedule(matchups, valid_teams)


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
    """Prompt user to enter bye teams and double header teams (manual fallback)."""
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

    return {
        'bye_teams': bye_teams,
        'double_header_teams': double_header_teams,
        'home_teams': [],
        'home_counts': {}
    }


if __name__ == "__main__":
    valid_teams = get_valid_teams()
    current_week = get_current_week()

    if current_week is not None:
        next_week = current_week + 1
        print(f"Current week in data: {current_week}")
        print(f"Fetching schedule for week {next_week}...\n")

        schedule_info = get_schedule_info(next_week, valid_teams)

        if schedule_info:
            print(f"\nSchedule detected:")
            print(f"  Bye teams: {schedule_info['bye_teams']}")
            print(f"  Double header teams: {schedule_info['double_header_teams']}")
            print(f"  Home teams: {schedule_info['home_teams']}")

            confirm = input("\nUse this schedule info? (y/n, default: y): ").strip().lower()
            if confirm != 'n':
                print()
                run_weekly_pipeline(
                    bye_teams=schedule_info['bye_teams'],
                    double_header_teams=schedule_info['double_header_teams'],
                    home_teams=schedule_info['home_teams'],
                    home_counts=schedule_info['home_counts']
                )
            else:
                print("\nUsing manual input instead...\n")
                info = prompt_for_teams()
                run_weekly_pipeline(**info)
        else:
            print("\nCould not parse schedule, using manual input...\n")
            info = prompt_for_teams()
            run_weekly_pipeline(**info)
    else:
        print("No existing scores data found. Running initial pipeline...\n")
        info = prompt_for_teams()
        run_weekly_pipeline(**info)
