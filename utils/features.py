import pandas as pd


def create_time_features(date, time):
    timestamp = pd.to_datetime(f"{date} {time}")

    features = {
        "Year": timestamp.year,
        "Month": timestamp.month,
        "Day": timestamp.day,
        "Hour": timestamp.hour,
        "DayOfWeek": timestamp.dayofweek,
        "IsWeekend": int(timestamp.dayofweek >= 5)
    }

    return features