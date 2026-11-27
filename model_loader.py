import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "model" / "random_forest_energy_model.pkl"
FEATURE_PATH = BASE_DIR / "model" / "energy_feature_names.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, features