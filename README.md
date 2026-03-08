To update weekly scores & lineup, run:

cd /Users/megan/Documents/Code\ Projects/fantasygym                                                
source .venv/bin/activate                                                                          
python main.py 

Description:

Problem: Create the highest-scoring team of college gymnasts based on expected score for a given week and athlete cost.

Solution: Complete pipeline to optimize the highest-scoring team within constraints based on predictive analytics. Scrapes weekly scores, schedule, and pricing information, cleans and restructures data, engineers features, predicts scores and likelihood to compete using ML models, and optimizes teams based on model output and custom algorithm.

Highlights: Takes advantage of market inefficiencies (static pricing across varied weekly conditions) to adjust the predicted score of gymnasts based on league differences in home vs away scores. Similarly, adjusts for double headers, in which the higher score across two competitions counts, by sampling a gymnast-specific distribution centered around the original predicted score. Tests various model types - linear regression, xgboost, and mixed effects - to select the one with the least error.
