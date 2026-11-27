import pandas as pd


def test_datetime_feature_generation():
    # Sample datetime
    datetime_value = pd.to_datetime("2024-01-01 14:00:00")

    # Generate time-based features
    year = datetime_value.year
    month = datetime_value.month
    day = datetime_value.day
    hour = datetime_value.hour
    day_of_week = datetime_value.dayofweek
    is_weekend = day_of_week >= 5

    # Verify feature values
    assert year == 2024
    assert month == 1
    assert day == 1
    assert hour == 14
    assert day_of_week == 0
    assert is_weekend is False


def test_weekend_detection():
    # Saturday
    saturday = pd.to_datetime("2024-01-06")

    # Sunday
    sunday = pd.to_datetime("2024-01-07")

    assert saturday.dayofweek >= 5
    assert sunday.dayofweek >= 5


def test_hour_feature():
    datetime_value = pd.to_datetime("2024-01-01 23:00:00")

    assert datetime_value.hour == 23