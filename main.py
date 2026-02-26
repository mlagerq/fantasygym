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


def run_weekly_pipeline(target_week, bye_teams=None, double_header_teams=None, home_teams=None, home_counts=None):
    """
    Run the complete weekly fantasy gymnastics pipeline.

    Args:
        target_week: Week number to predict
        bye_teams: List of team names with byes (gymnasts excluded from selection)
        double_header_teams: List of team names with double headers
        home_teams: List of team names competing at home
        home_counts: Dict of {team: number of home meets this week}
    """
    scores_file = "Files/scores_long_adjusted.csv"

    # Check if we already have score data
    skip_scrape = False
    if os.path.exists(scores_file):
        existing_df = pd.read_csv(scores_file)
        max_week = existing_df["Week"].max()
        if max_week >= target_week - 1:
            print(f"Score data already exists through week {int(max_week)}")
            response = input("Skip scraping and use existing data? (y/n, default: y): ").strip().lower()
            skip_scrape = response != 'n'

    if not skip_scrape:
        print("=" * 60)
        print("STEP 1: Scraping scores from Road to Nationals")
        print("=" * 60)
        scrape_scores()

        print("\n" + "=" * 60)
        print("STEP 2: Cleaning and transforming data")
        print("=" * 60)
        clean_data()
    else:
        print("\nSkipping scrape and clean steps, using existing data.")

    # Filter out any scores from target_week or later
    print(f"\nFiltering data to only include weeks before {target_week}...")
    df = pd.read_csv(scores_file)
    rows_before = len(df)
    df = df[df["Week"] < target_week]
    rows_after = len(df)
    if rows_before != rows_after:
        print(f"  Dropped {rows_before - rows_after} rows from week {target_week}+")
        df.to_csv(scores_file, index=False)
    else:
        print("  No rows to filter.")

    print("\n" + "=" * 60)
    print(f"STEP 3: Running predictions for week {target_week}")
    print("=" * 60)
    run_predictions(target_week=target_week)

    print("\n" + "=" * 60)
    print("STEP 4: Optimizing team selection")
    print("=" * 60)
    optimize_team(
        bye_teams=bye_teams,
        double_header_teams=double_header_teams,
        home_teams=home_teams,
        home_counts=home_counts,
        week=target_week
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

    # Prompt user for the current week number
    while True:
        week_input = input("Enter the current week number to predict: ").strip()
        try:
            current_week = int(week_input)
            if current_week < 1:
                print("Week number must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"\nWill scrape scores through week {current_week - 1}")
    print(f"Will scrape schedule for week {current_week}")
    print(f"Will predict scores for week {current_week}\n")

    schedule_info = get_schedule_info(current_week, valid_teams)

    if schedule_info:
        print(f"\nSchedule detected:")
        print(f"  Bye teams: {schedule_info['bye_teams']}")
        print(f"  Double header teams: {schedule_info['double_header_teams']}")
        print(f"  Home teams: {schedule_info['home_teams']}")

        confirm = input("\nUse this schedule info? (y/n/q to quit, default: y): ").strip().lower()
        if confirm == 'q':
            print("Exiting.")
            exit(0)
        elif confirm != 'n':
            dh_choice = input("Double headers: (u)se parsed, (i)gnore, or (c)ustom input? (u/i/c, default: u): ").strip().lower()
            if dh_choice == 'i':
                double_header_teams = []
            elif dh_choice == 'c':
                double_header_teams = prompt_for_team_list(
                    "Enter teams with DOUBLE HEADERS (comma-separated, or press Enter for none):",
                    valid_teams
                )
            else:
                double_header_teams = schedule_info['double_header_teams']
            print()
            run_weekly_pipeline(
                target_week=current_week,
                bye_teams=schedule_info['bye_teams'],
                double_header_teams=double_header_teams,
                home_teams=schedule_info['home_teams'],
                home_counts=schedule_info['home_counts']
            )
        else:
            print("\nUsing manual input instead...\n")
            info = prompt_for_teams()
            run_weekly_pipeline(target_week=current_week, **info)
    else:
        confirm = input("\nCould not parse schedule. Continue with manual input? (y/q to quit, default: y): ").strip().lower()
        if confirm == 'q':
            print("Exiting.")
            exit(0)
        info = prompt_for_teams()
        run_weekly_pipeline(target_week=current_week, **info)
