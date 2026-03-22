import pandas as pd
from scipy.stats import uniform , randint
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score , precision_score , recall_score


def train_delay_classification(df) :

    X = df.drop(["ARRIVAL_DELAY" , "DELAY"] , axis=1)
    Y = df["DELAY"]

    cat_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

    X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42 , stratify=Y)

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
        n_iter=25,
        cv=3,
        n_jobs=-1,
        random_state=42,
        scoring="f1"
    )

    weights = compute_sample_weight("balanced", Y_train)

    random_search.fit(X_train , Y_train , sample_weight=weights)

    model = random_search.best_estimator_

    return model , X_test , Y_test





def evaluate_classification(model , X_test , Y_test) :

    Y_pred = model.predict(X_test)

    # --------------------
    # Evaluation Metric
    # --------------------

    metric = {
        "F1 Score" : f1_score(Y_test , Y_pred) , 
        "Recall" : recall_score(Y_test , Y_pred) , 
        "Precision" : precision_score(Y_test , Y_pred)
    }

    return Y_pred , metric