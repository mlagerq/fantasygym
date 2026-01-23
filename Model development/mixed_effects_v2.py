#%% New model with mixed effects / partial pooling
### Format the data so only one score per row
import pandas as pd
import numpy as np

df = pd.read_csv("road_to_nationals_long.csv")
playerinfo = pd.read_csv("player_info.csv")

df = df.drop(df.columns[1], axis=1)
long_data = pd.merge(
    df, 
    playerinfo, 
    left_on=['GymnastID'], 
    right_on=['GymnastID'], 
    how='inner'
)
long_data = long_data.dropna(subset=['Score'])
long_data = long_data.dropna(subset=["MeetID"])

# Ensure sorted by GymnastID, Event, and Date
long_data = long_data.sort_values(by=["GymnastID", "Event", "Date"])

# Keep only gymnast-events with at least 3 scores in the season
df = (
    long_data.groupby(["GymnastID", "Event"], group_keys=False)
      .filter(lambda g: len(g) >= 3)
)

df['Gymnast_Event'] = df['GymnastID'].astype(str) + "_" + df['Event'].astype(str)

# %%
# Fit a mixed-effects model
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Response variable: Score
# Fixed effects: Week, Event, Team
model = smf.mixedlm("Score ~ Week + C(Event) + C(Team)", 
                               df, 
                               groups=df["GymnastID"],  
                               re_formula="1", # random intercept for gymnast
                               vc_formula={"MeetID": "0 + C(MeetID)"}  # Random effect for MeetID
 )
fit = model.fit()

# Gymnast event group
model2 = smf.mixedlm("Score ~ Week + C(Event) + C(Team)", 
                               df, 
                               groups=df["Gymnast_Event"],  
                               re_formula="1" # random intercept for gymnast_event
                               ,vc_formula={"MeetID": "0 + C(MeetID)"}  # Random effect for MeetID
 )
fit2 = model2.fit()

# Gymnast event group, no team effect
model3 = smf.mixedlm("Score ~ Week + C(Event)", 
                               df, 
                               groups=df["Gymnast_Event"],  
                               re_formula="1", # random intercept for gymnast_event
                               vc_formula={"MeetID": "0 + C(MeetID)"}  # Random effect for MeetID
 )
fit3 = model3.fit()

# Team as another group - cannot have multiple columns for groups.
# %%
print(fit.summary())
print('Gymnast variance:')
print(fit.cov_re)  # Random effect variance for GymnastID
print('Variance of gymnast and meet: ',fit.vcomp)  # Variance of both GymnastID and MeetID

# %%
from statsmodels.tools.eval_measures import rmse
y_actual = df['Score']

y_predicted = fit.predict()
model_rmse = rmse(y_actual, y_predicted)
print(f"RMSE: {model_rmse}")
# Gymnast group RMSE: .225

df['y2_predicted'] = fit2.predict()
model_rmse2 = rmse(y_actual, y2_predicted)
print(f"RMSE 2: {model_rmse2}")
# Gymnast-event group RMSE: .225

y3_predicted = fit3.predict()
model_rmse3 = rmse(y_actual, y3_predicted)
print(f"RMSE 3: {model_rmse3}")
# Gymnast-event group, no team RMSE: .234

#%%
# Plot the mean score vs mean predicted score for each gymnast/event
df2 = ( df.groupby("Gymnast_Event")
        .agg(
          score=('Score','mean'),
          predicted=('y2_predicted','mean'),
          count=('Score','size')
        )
        .reset_index()
)
df3 = df2[df2['count']>=8]
plt.scatter(df3["predicted"],df3["score"])

# %%
#Train test split by week with rolling origin

df = df.sort_values(["Week", "GymnastID", "Event"])
weeks = sorted(df["Week"].unique())

import statsmodels.formula.api as smf

#%% Fit models
results = []

# start at week 4, so we have at least 3 prior weeks of training data
for i in range(3, len(weeks)):
    train_weeks = weeks[:i]      # e.g., [1, 2, 3] → [1, 2, 3, 4] → ...
    test_week = weeks[i]         # predict this week
    
    train = df[df["Week"].isin(train_weeks)]
    test  = df[df["Week"] == test_week]
    
    # fit model
    model = smf.mixedlm(
        "Score ~ Week + C(Event) + C(Team)",
        train,
        groups=train["GymnastID"],
        re_formula="1",
        vc_formula={"MeetID": "0 + C(MeetID)"}
    )
    fit = model.fit(reml=False)
    
    # make predictions for test week
    test["Predicted"] = fit.predict(test)
    
    # store results
    results.append(test[["GymnastID", "Event", "Week", "Score", "Predicted"]])

cv_predictions = pd.concat(results, ignore_index=True)

#%%
from sklearn.metrics import mean_squared_error

#Calculate Rmse overall
rmse = mean_squared_error(cv_predictions["Score"], cv_predictions["Predicted"], squared=False)
print(f"Rolling-origin RMSE: {rmse:.3f}")

#%%
#Compare rmse by week
cv_predictions.groupby("Week").apply(
    lambda g: mean_squared_error(g["Score"], g["Predicted"], squared=False)
)

#TODO: Other options - add H/A as a fixed effect. Try interactions? 
#TODO: Logit