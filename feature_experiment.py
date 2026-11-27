import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "energy_prediction_app_data.csv"

FEATURE_PATH = BASE_DIR / "model" / "energy_feature_names.pkl"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

feature_names = joblib.load(FEATURE_PATH)

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
print("Creating additional features...")

# Additional lag features
df["Lag_3"] = df["Actual"].shift(3)
df["Lag_48"] = df["Actual"].shift(48)

# Rolling averages using only previous observations
df["Rolling_Mean_3"] = (
    df["Actual"].shift(1).rolling(window=3).mean()
)

df["Rolling_Mean_24"] = (
    df["Actual"].shift(1).rolling(window=24).mean()
)

# Remove rows with missing values caused by lag features
df = df.dropna().reset_index(drop=True)

# --------------------------------------------------
# BASELINE FEATURES
# --------------------------------------------------
baseline_features = feature_names

# --------------------------------------------------
# EXTENDED FEATURES
# --------------------------------------------------
additional_features = [
    "Lag_3",
    "Lag_48",
    "Rolling_Mean_3",
    "Rolling_Mean_24"
]

extended_features = baseline_features + additional_features

# --------------------------------------------------
# TARGET
# --------------------------------------------------
y = df["Actual"]

# --------------------------------------------------
# CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

y_train = train_df["Actual"]
y_test = test_df["Actual"]

# --------------------------------------------------
# BASELINE MODEL
# --------------------------------------------------
X_train_baseline = train_df[baseline_features]
X_test_baseline = test_df[baseline_features]

baseline_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining baseline model...")

baseline_model.fit(
    X_train_baseline,
    y_train
)

baseline_predictions = baseline_model.predict(
    X_test_baseline
)

# --------------------------------------------------
# FEATURE-ENGINEERED MODEL
# --------------------------------------------------
X_train_extended = train_df[extended_features]
X_test_extended = test_df[extended_features]

improved_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining feature-engineered model...")

improved_model.fit(
    X_train_extended,
    y_train
)

improved_predictions = improved_model.predict(
    X_test_extended
)

# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------
def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    return mae, rmse, r2

# --------------------------------------------------
# COMPARE MODELS
# --------------------------------------------------
baseline_results = evaluate_model(
    "BASELINE RANDOM FOREST",
    y_test,
    baseline_predictions
)

improved_results = evaluate_model(
    "FEATURE-ENGINEERED RANDOM FOREST",
    y_test,
    improved_predictions
)

# --------------------------------------------------
# COMPARISON
# --------------------------------------------------
print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame(
    {
        "Metric": ["MAE", "RMSE", "R2"],
        "Baseline": baseline_results,
        "Feature Engineered": improved_results
    }
)

print(comparison)

# --------------------------------------------------
# IMPROVEMENT CALCULATION
# --------------------------------------------------
mae_improvement = (
    (baseline_results[0] - improved_results[0])
    / baseline_results[0]
) * 100

print(
    f"\nMAE Improvement: {mae_improvement:.2f}%"
)

# --------------------------------------------------
# SAVE IMPROVED MODEL ONLY IF MAE IS BETTER
# --------------------------------------------------
if improved_results[0] < baseline_results[0]:

    improved_model_path = (
        BASE_DIR / "model" / "feature_engineered_rf_model.pkl"
    )

    improved_feature_path = (
        BASE_DIR / "model" / "feature_engineered_names.pkl"
    )

    joblib.dump(
        improved_model,
        improved_model_path
    )

    joblib.dump(
        extended_features,
        improved_feature_path
    )

    print("\nImproved model saved successfully!")
    print(improved_model_path)

else:

    print(
        "\nThe feature-engineered model did not improve MAE."
    )

    print(
        "The baseline model will be retained."
    )