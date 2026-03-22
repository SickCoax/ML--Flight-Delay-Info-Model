import pandas as pd


def preprocess(df) :

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

    df = df.drop(cols_to_remove , axis=1 , errors="ignore")

    df = df.dropna(subset=["ARRIVAL_DELAY"])

    cat_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

    for col in cat_cols:
        df[col] = df[col].astype(str)

    df["SCHEDULED_DEPARTURE_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
    df["SCHEDULED_ARRIVAL_HOUR"] = df["SCHEDULED_ARRIVAL"] // 100

    df = df.drop(["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"], axis=1)

    df["DELAY"] = (df["ARRIVAL_DELAY"] >=15).astype(int)

    return df