#%%
# OLD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# 📌 Load the dataset
# one row per score
# home scores already adjusted based on league homeaway factor
df = pd.read_csv("scores_long_adjusted.csv")

X_train = df[df['Week'] < 8]
X_test  = df[df['Week'] == 8]

# 📌 Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Print Coefficients
coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_})
print("\n📊 Linear Regression Coefficients:")
print(coefficients)

# 📌 Print Intercept
print(f"\n📊 Intercept (Baseline Prediction): {model.intercept_:.3f}")

# 📌 Make predictions
df_model['predicted_week8_score'] = model.predict(df_model[X_train.columns])
# %%
## Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(df_model["Score_8"], df_model["predicted_week8_score"])
mse = mean_squared_error(df_model["Score_8"], df_model["predicted_week8_score"])
rmse = mse ** 0.5

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

#MAE: 0.1003
#RMSE: 0.1874

# %%
# Examine residuals
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Calculate residuals
df_model["Residuals"] = df_model["Score_8"] - df_model["predicted_week8_score"]

# Histogram of Residuals
plt.figure(figsize=(8, 5))
sns.histplot(df_model["Residuals"], bins=30, kde=True)
plt.axvline(0, color='red', linestyle='dashed')  
plt.title("Histogram of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Residuals vs. Actual Scores
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_model["Score_8"], y=df_model["Residuals"], alpha=0.6)
plt.axhline(0, color="red", linestyle="dashed")  # Reference line at 0
plt.title("Residuals vs. Actual Scores")
plt.xlabel("Actual Standardized Score (Score_Z)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.show()
### positive slope




#STOP HERE FOR LINEAR





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

# average scores within the week if they compete twice 
agg_data = (
    long_data.groupby(["GymnastID", "Week", "Event", "Team"], as_index=False)
    .agg({"Score": "mean"})
)

# Save this off so we can use it in other models
agg_data.to_csv("mixed_effects_model_input.csv", index=False)
long_data.to_csv("mixed_effects_model_input_w_meetid.csv", index=False)

# %%

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit a mixed-effects model
# model_no_meet = smf.mixedlm("Score ~ Week + Team + Event", 
#                             agg_data, 
#                             groups=agg_data["GymnastID"])
#fit_no_meet = model_no_meet.fit()
#print(fit_no_meet.summary())

model_with_meet = smf.mixedlm("Score ~ Week + C(Event) + C(Team)", 
                               df_clean, 
                               groups=df_clean["GymnastID"],  
                               re_formula="1", # random intercept for gymnast
                               vc_formula={"MeetID": "0 + C(MeetID)"}  # Random effect for MeetID
 )
fit_with_meet = model_with_meet.fit()
#print(fit_with_meet.summary())

# Compare variance with and without meet variable
#print("Variance (GymnastID only):")
#print(fit_no_meet.cov_re)  # Random effect variance for GymnastID
# 0.00947
#print("Variance components:")
#print(fit_with_meet.vcomp)  # Variance of both GymnastID and MeetID
# 0.03

# %%
### Gymnast effects are very small and getting overshadowed. try standardizing scores since they are clumped around 9.75
mean_score = agg_data["Score"].mean()
std_score = agg_data["Score"].std()
agg_data["Score_Z"] = (agg_data["Score"] - mean_score) / std_score

#%%
# Training data: all weeks before week 8
train_data = agg_data[agg_data["Week"] < 8]

# Test data: only week 8
test_data = agg_data[agg_data["Week"] == 8]

#%%
### With team
model_z = smf.mixedlm(
    "Score_Z ~ Week + C(Team) + C(Event)",
    train_data,
    groups=train_data["GymnastID"],
    re_formula="1" 
)

fit_z = model_z.fit(reml=True)
print(fit_z.summary())
test_data["Predicted_Score"] = fit_z.predict(test_data)
test_data.to_csv("z_score_model_pred.csv", index=False)

# gymnastID variance (group var) = 0.133

### Without team
model_no_team = smf.mixedlm(
    "Score_Z ~ Week + C(Event)",
    train_data,
    groups=train_data["GymnastID"],
    re_formula="1"
)
fit_no_team = model_no_team.fit(reml=True)
print(fit_no_team.summary())
test_data["Predicted_Score_No_Team"] = fit_no_team.predict(test_data)
# gymnastID variance (group var) = 0.255

## Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_data["Score_Z"], test_data["Predicted_Score"])
mse = mean_squared_error(test_data["Score_Z"], test_data["Predicted_Score"])
rmse = mse ** 0.5

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

mae = mean_absolute_error(test_data["Score_Z"], test_data["Predicted_Score_No_Team"])
mse = mean_squared_error(test_data["Score_Z"], test_data["Predicted_Score_No_Team"])
rmse = mse ** 0.5

print(f"MAE 2: {mae:.4f}")
print(f"RMSE 2: {rmse:.4f}")

### Without team but with week 
#model_no_team = smf.mixedlm(
#    "Score_Z ~ Week + C(Event)",
#    train_data,
#    groups=train_data["GymnastID"],
#    re_formula="Week"
    #assume that gymnasts have a random intercept 
    # and different improvement slopes by week
#)
#fit_no_team = model_no_team.fit(reml=True)
#print(fit_no_team.summary())
## Group x Week Cov = -0.022, Week Var= 0.002
## Interpretation: week variance is very small, higher scoring gymnasts improve less week-by-week

#%%
### try logit transformation of scores since there is a ceiling at 10.0
### jk don't do logit
### start here
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("mixed_effects_model_input_w_meetid.csv")

## Filter scores in range [9, 10]
#df = df[df["Score"] >= 9.0]

#epsilon = 1e-5  # Small adjustment to avoid exact 0 or 1
#df["Score_Adjusted"] = np.clip(df["Score"], 9.0001, 9.9999)

## Apply logit transform to (Score - 9), ensuring values stay within (0,1)
#df["Score_Logit"] = np.log((df["Score_Adjusted"] - 9) / (10 - df["Score_Adjusted"]))

## Training data: all weeks before week 8
train_data = df[df["Week"] < 9]

## Test data: only week 8
test_data = df[df["Week"] == 9]

test_data['Gymnast_Event'] = test_data['GymnastID'].astype(str) + "_" + test_data['Event'].astype(str)
train_data['Gymnast_Event'] = train_data['GymnastID'].astype(str) + "_" + train_data['Event'].astype(str)
test_data = test_data[test_data['Gymnast_Event'].isin(train_data['Gymnast_Event'])]

#%%
### With team
model_1 = smf.mixedlm(
    #"Score_Logit ~ Week + C(Team) + C(Event)",
    #"Score ~ Week + C(Team) + C(Event)",
    "Score ~ Week",
    train_data,
    groups=train_data["GymnastID"],
    re_formula="1" 
)

fit_1 = model_1.fit(reml=True)
print(fit_1.summary())
test_data["Predicted_Score"] = fit_1.predict(test_data)
# group var 0.088

#back transform
#test_data["Predicted_Score_LBack"] = (10 * np.exp(test_data["Predicted_Score_Logit"]) + 9) / (1 + np.exp(test_data["Predicted_Score_Logit"]))
#test_data.to_csv('logit_score_model_pred.csv',index=False)

#%%
# Use Gymnast_Event combo as grouping and MeetID as random effect
model = smf.mixedlm(
    #"Score ~ Week + C(Event) * C(Team)",  # team-event interaction
    "Score ~ Week", 
    train_data,
    groups=train_data["Gymnast_Event"], 
    re_formula="1",
    vc_formula={"MeetID": "0 + C(MeetID)"}  # Random effect for MeetID
)

fit = model.fit(reml=True)
print(fit.summary())
test_data["Predicted_Score_team"] = fit.predict(test_data)
# group var 0.22

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_data["Score"], test_data["Predicted_Score"])
mse = mean_squared_error(test_data["Score"], test_data["Predicted_Score"])
rmse = mse ** 0.5

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

""" 
Interaction team x event, gymnast
MAE: 0.1185
RMSE: 0.2376

With meet ID, no interactions, no gymnast fixed effect
MAE: 0.1070
RMSE: 0.1984 
"""

#%%
pricing = pd.read_csv("fantasizr_player_pricing.csv")
final_df = pd.merge(
    test_data[["GymnastID", "Name", "Event", "Team", "Week", "Predicted_Score"]], 
    pricing[['Player Name','Event','Price']], 
    left_on=['Name', 'Event'], 
    right_on=['Player Name', 'Event'], 
    how='inner'
)
final_df = final_df.drop('Player Name',axis=1)
final_df.to_csv('team_opt_input_mixed.csv',index=False)

# %%
# Examine residuals
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Calculate residuals
test_data["Residuals"] = test_data["Score"] - test_data["Predicted_Score"]

# Histogram of Residuals
plt.figure(figsize=(8, 5))
sns.histplot(test_data["Residuals"], bins=30, kde=True)
plt.axvline(0, color='red', linestyle='dashed')  
plt.title("Histogram of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Residuals vs. Predicted Scores
plt.figure(figsize=(8, 5))
sns.scatterplot(x=test_data["Predicted_Score"], y=test_data["Residuals"], alpha=0.6)
plt.axhline(0, color="red", linestyle="dashed")  # Reference line at 0
plt.title("Residuals vs. Predicted Scores")
plt.xlabel("Predicted Score")
plt.ylabel("Residuals (Actual - Predicted)")
plt.show()

#%%
# Scatter plot: Predicted vs. Actual Scores
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.figure(figsize=(8, 5))
sns.scatterplot(x=test_data["Predicted_Score"], y=test_data[test_data["Score"]>=9.5]["Score"], alpha=0.6)
plt.axhline(9.5, color="red", linestyle="dashed")  # Reference line at 0
plt.title("Actual vs. Predicted Scores")
plt.xlabel("Predicted Score")
plt.ylabel("Actual Score")
plt.show()

#%%

# Define bins from 9.5 to 10.0 with a step of 0.05
bins = np.arange(9.525, 10.025, 0.05)
labels = [round(b + 0.025, 3) for b in bins[:-1]]  # Midpoint labels for bins

# Assign predicted scores to bins
test_data['bin'] = pd.cut(test_data['Predicted_Score'], bins=bins, labels=labels, include_lowest=True)

# Group by bin and compute mean residual
bin_means = test_data.groupby('bin')['Residuals'].mean().reset_index()

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=bin_means['bin'], y=bin_means['Residuals'], color='skyblue')

plt.xlabel('Predicted Score Bucket')
plt.ylabel('Average Residual')
plt.title('Average Residual Across Predicted Score Buckets')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line at 0
plt.show()


# Group by bin and compute mean residual and sample size
bin_stats = test_data.groupby('bin').agg(mean_residual=('Residuals', 'mean'), sample_size=('Residuals', 'count')).reset_index()

# Plot setup
fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar plot for mean residual
sns.barplot(x=bin_stats['bin'], y=bin_stats['mean_residual'], color='skyblue', ax=ax1)

# Labels
ax1.set_xlabel('Predicted Score Bucket')
ax1.set_ylabel('Average Residual', color='blue')
ax1.set_title('Average Residual Across Predicted Score Buckets')

# Annotate sample size on top of bars
for i, (bin_label, residual, count) in enumerate(zip(bin_stats['bin'], bin_stats['mean_residual'], bin_stats['sample_size'])):
    ax1.text(i, residual + 0.001, str(count), ha='center', va='bottom', fontsize=10, color='black')

# Add reference line at 0
ax1.axhline(0, color='black', linestyle='--', linewidth=1)

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

plt.show()

#%%
## Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_data["Score"], test_data["Predicted_Score_Logit"])
mse = mean_squared_error(test_data["Score"], test_data["Predicted_Score_Logit"])
rmse = mse ** 0.5

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

mae = mean_absolute_error(test_data["Score"], test_data["Predicted_Score_No_Team_Logit"])
mse = mean_squared_error(test_data["Score"], test_data["Predicted_Score_No_Team_Logit"])
rmse = mse ** 0.5

print(f"MAE 2: {mae:.4f}")
print(f"RMSE 2: {rmse:.4f}")

#MAE: 0.1249
#RMSE: 0.2349
#MAE 2: 0.1343
#RMSE 2: 0.2406
#%%
## Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_data["Score_Logit"], test_data["Predicted_Score_Logit"])
mse = mean_squared_error(test_data["Score_Logit"], test_data["Predicted_Score_Logit"])
rmse = mse ** 0.5

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

mae = mean_absolute_error(test_data["Score_Logit"], test_data["Predicted_Score_No_Team_Logit"])
mse = mean_squared_error(test_data["Score_Logit"], test_data["Predicted_Score_No_Team_Logit"])
rmse = mse ** 0.5

print(f"MAE 2: {mae:.4f}")
print(f"RMSE 2: {rmse:.4f}")

# %%

df = pd.read_csv("z_score_model_pred.csv")
df2 = pd.read_csv("logit_score_model_pred.csv")

df["Predicted_Score_ZBack"] = (df["Predicted_Score"] * std_score) + mean_score
df2["Predicted_Score_LBack"] = (10 * np.exp(df2["Predicted_Score_Logit"]) + 9) / (1 + np.exp(df2["Predicted_Score_Logit"]))
df2["Predicted_Score_LBack_NT"] = (10 * np.exp(df2["Predicted_Score_No_Team_Logit"]) + 9) / (1 + np.exp(df2["Predicted_Score_No_Team_Logit"]))

comb_df = pd.merge(
    df[['GymnastID','Week','Event','Team','Score','Predicted_Score_ZBack']], 
    df2[['GymnastID','Week','Event','Predicted_Score_LBack']], 
    left_on=['GymnastID','Week','Event'], 
    right_on=['GymnastID','Week','Event'], 
    how='inner'
)
comb_df["Residual_Z"] = comb_df["Score"] - comb_df["Predicted_Score_ZBack"]
comb_df["Residual_Logit"] = comb_df["Score"] - comb_df["Predicted_Score_LBack"]

# %%
# Plots!
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(comb_df["Residual_Z"], color="blue", label="Z-score Model", kde=True, bins=30)
sns.histplot(comb_df["Residual_Logit"], color="red", label="Logit Model", kde=True, bins=30)
sns.histplot(df_model["Residuals"], color="green", label="Linear Model", kde=True, bins=30)
plt.axvline(0, color='black', linestyle='dashed', linewidth=1)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 5))
plt.scatter(comb_df["Score"], comb_df["Residual_Z"], color="blue", alpha=0.5, label="Z-score Model")
plt.scatter(comb_df["Score"], comb_df["Residual_Logit"], color="red", alpha=0.5, label="Logit Model")
plt.scatter(df_model["Score_8"], df_model["Residuals"], color="green", alpha=0.5, label="Linear Model")
plt.axhline(0, color='black', linestyle='dashed', linewidth=1)
plt.xlabel("Actual Score")
plt.ylabel("Residual")
plt.title("Residuals vs Actual Scores")
plt.legend()
plt.show()

#%%
## Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(df["Score"],df["Predicted_Score_ZBack"])
mse = mean_squared_error(df["Score"], df["Predicted_Score_ZBack"])
rmse = mse ** 0.5

print(f"MAE (Z): {mae:.4f}")
print(f"RMSE (Z): {rmse:.4f}")


mae = mean_absolute_error(df2["Score"], df2["Predicted_Score_LBack"])
mse = mean_squared_error(df2["Score"], df2["Predicted_Score_LBack"])
rmse = mse ** 0.5

print(f"MAE (L): {mae:.4f}")
print(f"RMSE (L): {rmse:.4f}")

#No team effect
mae = mean_absolute_error(df2["Score"], df2["Predicted_Score_LBack_NT"])
mse = mean_squared_error(df2["Score"], df2["Predicted_Score_LBack_NT"])
rmse = mse ** 0.5

print(f"MAE (L): {mae:.4f}")
print(f"RMSE (L): {rmse:.4f}")

##  the logit model histogram is centered closer to 0 but still a bit positive. The slope of both is positive
## options to improve on logit:
#### adjusted logit
#### beta regression
#### non-linear week effect (use spline term)
## also still need to look at residuals without team
## next: xgboost? bayesian?
# %%
# linear regression
##MAE: 0.1003
##RMSE: 0.1874
# mixed effects
##MAE (Z): 0.1249
##RMSE (Z): 0.2349
##MAE (L): 0.0886
##RMSE (L): 0.1487
# mixed effects no team
##MAE (L): 0.0965
##RMSE (L): 0.1550