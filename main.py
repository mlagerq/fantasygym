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


def run_weekly_pipeline():
    """Run the complete weekly fantasy gymnastics pipeline."""
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
    optimize_team()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("Check lineup.csv for your optimized team")
    print("=" * 60)


if __name__ == "__main__":
    run_weekly_pipeline()
