import pandas as pd
from scipy.stats import uniform , randint
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error , mean_absolute_error

def train_delay_regression(df) :

    X = df.drop(["ARRIVAL_DELAY", "DELAY"] , axis=1)
    Y = df["ARRIVAL_DELAY"]

    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)

    cat_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    
    preprocess = ColumnTransformer([("cat" , OrdinalEncoder() , cat_cols)] , remainder="passthrough")

    pipeline = Pipeline([("preprocess" , preprocess) , ("GBR" , GradientBoostingRegressor())])


    param = {
        "GBR__n_estimators": randint(100, 400),
        "GBR__learning_rate": uniform(0.03, 0.2),
        "GBR__max_depth": randint(3, 8),
        "GBR__subsample": uniform(0.6, 0.4),
        "GBR__min_samples_split": randint(2, 8),
        "GBR__min_samples_leaf": randint(1, 6)
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param,
        n_iter=25,
        cv=3,
        n_jobs=-1,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    random_search.fit(X_train , Y_train)

    model = random_search.best_estimator_

    return model , X_test , Y_test




def evaluate_regression(model , X_test , Y_test):

    Y_pred = model.predict(X_test)

    # --------------------
    # Evaluation Metric
    # --------------------

    metric = {
        "MSE" : mean_squared_error(Y_test , Y_pred) , 
        "MAE" : mean_absolute_error(Y_test , Y_pred)
    }

    return Y_pred , metric