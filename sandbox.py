# This file contains experimental code and development iterations and does not made for execution .
# Final structured pipeline is in the /src folder.

import pandas as pd
from scipy.stats import uniform , randint
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor , GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error , mean_absolute_error , f1_score , recall_score , precision_score , confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

df = pd.read_csv(r"dataset/flights.csv")

df = df[df["CANCELLED"] == 0]

cols_to_remove = [
"YEAR",
"FLIGHT_NUMBER",
"TAIL_NUMBER",
"ARRIVAL_TIME",
"DEPARTURE_TIME",
"DEPARTURE_DELAY",
"TAXI_OUT",
"WHEELS_OFF",
"WHEELS_ON",
"TAXI_IN",
"AIR_TIME",
"ELAPSED_TIME",
"DIVERTED",
"CANCELLED",
"CANCELLATION_REASON",
"AIR_SYSTEM_DELAY",
"SECURITY_DELAY",
"AIRLINE_DELAY",
"LATE_AIRCRAFT_DELAY",
"WEATHER_DELAY"
]

df = df.drop(cols_to_remove , axis=1)

df = df.dropna(subset=["ARRIVAL_DELAY"])

cat_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

for col in cat_cols:
    df[col] = df[col].astype(str)

df["SCHEDULED_DEPARTURE_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
df["SCHEDULED_ARRIVAL_HOUR"] = df["SCHEDULED_ARRIVAL"] // 100

df = df.drop(["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"], axis=1)

df["DELAY"] = (df["ARRIVAL_DELAY"] >=15).astype(int)

# --------------------
# --------------------
# Regression Work
# --------------------
# --------------------

# x = df.drop(["ARRIVAL_DELAY" , "DELAY"] , axis=1)
# y = df["ARRIVAL_DELAY"]

# xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size=0.2 , random_state=42)


# preprocess = ColumnTransformer([("cat" , OrdinalEncoder() , cat_cols)] , remainder="passthrough")

# pipeline = Pipeline([("preprocess" , preprocess) , ("GBR" , GradientBoostingRegressor())])

# param = {
#     "GBR__n_estimators": randint(100, 400),
#     "GBR__learning_rate": uniform(0.03, 0.2),
#     "GBR__max_depth": randint(3, 8),
#     "GBR__subsample": uniform(0.6, 0.4),
#     "GBR__min_samples_split": randint(2, 8),
#     "GBR__min_samples_leaf": randint(1, 6)
# }

# random_search = RandomizedSearchCV(
#     pipeline,
#     param,
#     n_iter=15,
#     cv=3,
#     n_jobs=-1,
#     random_state=42,
#     scoring="neg_mean_absolute_error"
# )

# random_search.fit(xtrain , ytrain)

# model = random_search.best_estimator_

# ypred = model.predict(xtest)


# # --------------------
# # Evaluation Metric
# # --------------------

# print(f"MSE : {mean_squared_error(ytest , ypred)}")
# print(f"MAE : {mean_absolute_error(ytest , ypred)}")



# --------------------
# --------------------
# Claasification Work
# --------------------
# --------------------

x = df.drop(["ARRIVAL_DELAY" , "DELAY"] , axis=1)
y = df["DELAY"]

X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size=0.2 , random_state=42 , stratify=y)

preprocess = ColumnTransformer([("cat" , OrdinalEncoder() , cat_cols)] , remainder="passthrough")

pipeline = Pipeline([("preprocess" , preprocess) , ("GBC" , GradientBoostingClassifier())])

param = {
    "GBC__n_estimators": randint(100, 400),
    "GBC__learning_rate": uniform(0.03, 0.2),
    "GBC__max_depth": randint(3, 8),
    "GBC__subsample": uniform(0.6, 0.4),
    "GBC__min_samples_split": randint(2, 8),
    "GBC__min_samples_leaf": randint(1, 6)
}

random_search = RandomizedSearchCV(
    pipeline,
    param,
    n_iter=15,
    cv=3,
    n_jobs=-1,
    random_state=42,
    scoring="f1"
)

weights = compute_sample_weight("balanced", Y_train)

random_search.fit(X_train , Y_train , sample_weight=weights)

model = random_search.best_estimator_

ypred = model.predict(X_test)


# --------------------
# Evaluation Metric
# --------------------

print(f"F1 Score : {f1_score(Y_test , ypred)}")
print(f"Recall : {recall_score(Y_test , ypred)}")
print(f"Precison : {precision_score(Y_test , ypred)}")
print(f"Confusion Matrix :\n {confusion_matrix(Y_test , ypred)}")