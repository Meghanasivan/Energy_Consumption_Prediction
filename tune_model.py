
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "energy_prediction_app_data.csv"

MODEL_PATH = BASE_DIR / "model" / "random_forest_energy_model.pkl"

FEATURE_PATH = BASE_DIR / "model" / "energy_feature_names.pkl"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# Load feature names used by the original model
feature_names = joblib.load(FEATURE_PATH)

X = df[feature_names]
y = df["Actual"]

# --------------------------------------------------
# CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# RANDOM FOREST HYPERPARAMETER TUNING
# --------------------------------------------------
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4]
}

tscv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1,
    verbose=1
)

print("\nStarting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------
best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation MAE:")
print(-grid_search.best_score_)

# --------------------------------------------------
# TEST SET EVALUATION
# --------------------------------------------------
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== TUNED MODEL RESULTS ==========")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# --------------------------------------------------
# SAVE TUNED MODEL SEPARATELY
# --------------------------------------------------
TUNED_MODEL_PATH = BASE_DIR / "model" / "tuned_random_forest_model.pkl"

joblib.dump(best_model, TUNED_MODEL_PATH)

print("\nTuned model saved successfully!")
print(TUNED_MODEL_PATH)
