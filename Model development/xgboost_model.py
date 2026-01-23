#%%
# Add features
import pandas as pd
import numpy as np

df = pd.read_csv("road_to_nationals_long.csv")

# Sort by GymnastID, Event, and Date to ensure chronological order
df = df.sort_values(by=["GymnastID", "Event", "Date"])

# Function to compute rolling statistics up to each row's date
def compute_rolling_features(group):
    group["High_Score"] = group["Score"].expanding().max().shift(1)
    group["Low_Score"] = group["Score"].expanding().min().shift(1)
    group["Average"] = group["Score"].expanding().mean().shift(1)
    group["Last_3_raw"] = (
        4 * group["Score"].shift(1) + 2 * group["Score"].shift(2) + group["Score"].shift(3)
    )
    group["Last_3"] = group["Last_3_raw"]/7
    group["score_1"] = group["Score"].shift(1)
    group["score_2"] = group["Score"].shift(2)
    group["score_3"] = group["Score"].shift(3)
    return group

# Apply function grouped by Gymnast and Event
df = df.groupby(["GymnastID", "Event"], group_keys=False).apply(compute_rolling_features)

# Drop rows where we don't have enough history
df = df.dropna(subset=["Last_3"])
df = df.drop('Last_3_raw',axis=1)

#%%
players = pd.read_csv("player_info.csv")
df = pd.merge(
    df, 
    players, 
    left_on=['GymnastID'], 
    right_on=['GymnastID'], 
    how='inner'
)
df = df.drop('MeetID', axis=1)
df = df.dropna()  # Drop NaNs
df = df.replace([np.inf, -np.inf], np.nan).dropna()  # Drop infinities

# %%
# xgboost model
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# One-hot encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cols = encoder.fit_transform(df[["Event", "HomeAway"]])   #,"Team", "GymnastID"]])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(["Event", "HomeAway"])) #"Team", "GymnastID"]))

# Merge encoded columns back
df = df.reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

# Select features
#feature_cols = encoder.get_feature_names_out(["Team", "GymnastID"]).tolist() + ["Week", "High_Score", "Average", "Last_3"]
feature_cols = encoder.get_feature_names_out(["Event", "HomeAway"]).tolist() + ["Week", "High_Score", "Low_Score", "Average", "score_1", "score_2", "score_3"]
X = df[feature_cols]
y = df["Score"]  # Assuming Score is the target variable

#%%
# Create parameter tuning grid
params = {
        'min_child_weight': [5, 10, 15],
        'gamma': [0, 0.05, 0.1, 0.18, 0.25, 0.5, 1],
        'subsample': [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6]
        }
    
# Set up XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=5000, learning_rate=0.02)

# Set up splits
#### Ideally we should not have the same gymnasts in the train and test - is there a param for this? 
folds = 5
param_comb = 100
skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)
random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=param_comb, scoring='neg_root_mean_squared_error', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )

# Here we go
random_search.fit(X, y)
best_model = random_search.best_estimator_

print(random_search.best_params_) 
print(random_search.best_score_)
#{'subsample': 1.0, 'min_child_weight': 10, 'max_depth': 5, 'gamma': 0.5, 'colsample_bytree': 0.8}

#%%
# Evaluate feature importance
import shap
# Create a SHAP explainer object
explainer = shap.TreeExplainer(best_model)

# Calculate SHAP values for your data (e.g., test set)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)

#%%
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean

linear_model = LinearRegression()
skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)
scores = cross_val_score(linear_model, X, y, scoring='neg_root_mean_squared_error',
                         cv=skf, n_jobs=-1)
mean(scores) #-0.195

#%%
linear_model = LinearRegression()
linear_model.fit(X,y)

import statsmodels.api as sm

x = sm.add_constant(X)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())
# %%
# to run just once

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Square root of MSE

print(f"MAE: {mae:.4f}") # MAE: 0.104
print(f"RMSE: {rmse:.4f}") # RMSE: 0.195

# %%
# Predict next week 
current_week = 10
next_week = 11
next_week_features = df[df["Week"] == current_week].copy()  # Take only last week's data
next_week_features["Week"] = next_week  # Change to next week
next_week_features = next_week_features.drop('Score', axis=1)
next_week_features_og = next_week_features.copy()

missing_cols = set(X_train.columns) - set(next_week_features.columns)
for col in missing_cols:
    next_week_features[col] = 0  # Add missing columns with 0


next_week_features = next_week_features[X_train.columns]  # Ensure same order as training

# %%
y_pred = xgb_model.predict(next_week_features)  
next_week_features_og["Predicted_Score"] = y_pred
next_week_features_og[["GymnastID", "Name", "Event", "Team", "Week", "Predicted_Score"]]

# %%
pricing = pd.read_csv("fantasizr_player_pricing.csv")
final_df = pd.merge(
    next_week_features_og[["GymnastID", "Name", "Event", "Team", "Week", "Predicted_Score"]], 
    pricing[['Player Name','Event','Price']], 
    left_on=['Name', 'Event'], 
    right_on=['Player Name', 'Event'], 
    how='inner'
)
final_df = final_df.drop('Player Name',axis=1)
final_df.to_csv('team_opt_input.csv',index=False)
# %%
