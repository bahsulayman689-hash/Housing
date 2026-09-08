#-------------------------------------------------------------------------------------
#-------------------------importing require dependencies------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


import joblib
import pickle

import warnings as wr
wr.filterwarnings("ignore")
#-------------------------------------------------------------------------


#-----------------------------------------------------------------------------------
#---------------------------loading the data and inspecting the data----------------
data = pd.read_csv("model-test/data.csv")
print(data.head())
print(data.tail())
print(data.shape)
print(data.info())
print(data.isnull().sum() / len(data)*100)
print(data.duplicated().sum())

#-------------------------------------------------------------------------------


#----------------------------------------------------------------------
#-----------------------------spliting the data in test and train-------------------
X = data.drop(columns=["price", "date", "street", "city", "statezip", "country"])
print(X)
print(X.shape)
print(X.isnull().sum())
Y = data["price"]
print(Y)
print(Y.shape)
print(Y.isnull().sum())
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train.shape)
print(X_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3))
])
pram_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],   
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

grid_Search = GridSearchCV(estimator=pipeline, param_grid=pram_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_Search.fit(X_train, Y_train)


print("Best Parameters:", grid_Search.best_params_)
print("Best Score:", grid_Search.best_score_)

best_model = grid_Search.best_estimator_
test_predictions = best_model.predict(X_test)
mse = mean_squared_error(Y_test, test_predictions)
r2 = r2_score(Y_test, test_predictions)
print("Test MSE:", mse)
print("Test R2 Score:", r2)

model = joblib.dump(best_model, "model-test/best_model.pkl")
model = joblib.load("model-test/best_model.pkl")