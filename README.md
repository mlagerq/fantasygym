To update weekly scores & lineup, run:

cd /Users/megan/Documents/Code\ Projects/fantasygym                                                
source .venv/bin/activate                                                                          
python main.py 

-------

Problem: Create the highest-scoring team of college gymnasts based on expected score for a given week and athlete cost.

Solution: Complete pipeline to optimize the highest-scoring team within constraints based on predictive analytics. Scrapes weekly scores, schedule, and pricing information, cleans and restructures data, engineers features, predicts scores and likelihood to compete using ML models, and optimizes teams based on model output and custom algorithm. The core model predicts the next week's score based on the week number and the athlete's high score, most recent scores, and average score this season using linear regression.

Highlights: Takes advantage of market inefficiencies (static pricing across varied weekly conditions) to adjust the predicted score of gymnasts based on league differences in home vs away scores. Similarly, adjusts for double headers, in which the higher score across two competitions counts, by sampling a gymnast-specific distribution centered around the original predicted score. Various features and model types - linear regression, xgboost, and mixed effects - were tested before selecting the one with the least error.

Limtations: Models are trained only on one season of data - additional data would likely improve performance. Data is particuarly sparse early in the season - data from past seasons for returning athletes could possibly be integrated to make the model more robust. The team optimizer algorithm is highly specific to the fantasy league I participate in, but similar techniques could be applied to new problems.
