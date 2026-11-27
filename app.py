import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

from model_loader import load_model
from database.database import initialize_database, save_prediction

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "energy_prediction_app_data.csv"

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("⚡ Household Energy Consumption Prediction")

st.write(
    "Predict household electricity consumption using "
    "a trained Random Forest Regression model."
)

st.divider()

# --------------------------------------------------
# INITIALIZE DATABASE
# --------------------------------------------------
initialize_database()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()

model, feature_names = get_model()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------
st.success("Random Forest model loaded successfully!")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Dataset Records", f"{len(df):,}")

with col3:
    st.metric("Model Features", len(feature_names))

st.divider()

# --------------------------------------------------
# PREDICTION INTERFACE
# --------------------------------------------------
st.header("🔮 Energy Consumption Prediction")

st.write(
    "Select a date and time from the available historical dataset "
    "to generate a prediction."
)

# Get available dates
available_dates = sorted(df["Datetime"].dt.date.unique())

selected_date = st.date_input(
    "Select Date",
    min_value=min(available_dates),
    max_value=max(available_dates),
    value=available_dates[0]
)

# Filter data for selected date
date_data = df[df["Datetime"].dt.date == selected_date]

if not date_data.empty:

    available_hours = sorted(
        date_data["Datetime"].dt.hour.unique()
    )

    selected_hour = st.selectbox(
        "Select Hour",
        available_hours,
        format_func=lambda hour: f"{hour:02d}:00"
    )

    selected_row = date_data[
        date_data["Datetime"].dt.hour == selected_hour
    ]

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    if st.button("⚡ Predict Energy Consumption", type="primary"):

        if not selected_row.empty:

            row = selected_row.iloc[0]

            # Prepare model input
            input_data = pd.DataFrame(
                [row[feature_names].values],
                columns=feature_names
            )

            # Generate prediction
            prediction = float(model.predict(input_data)[0])

            # Get actual consumption
            actual = (
                float(row["Actual"])
                if "Actual" in row.index and pd.notna(row["Actual"])
                else None
            )

            # Calculate prediction error
            error = (
                abs(prediction - actual)
                if actual is not None
                else None
            )

            # --------------------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------------------
            st.subheader("Prediction Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Predicted Consumption",
                    f"{prediction:.4f} kW"
                )

            with result_col2:
                if actual is not None:
                    st.metric(
                        "Actual Consumption",
                        f"{actual:.4f} kW"
                    )
                else:
                    st.metric(
                        "Actual Consumption",
                        "Not available"
                    )

            with result_col3:
                if error is not None:
                    st.metric(
                        "Absolute Prediction Error",
                        f"{error:.4f} kW"
                    )
                else:
                    st.metric(
                        "Prediction Error",
                        "Not available"
                    )

            # --------------------------------------------------
            # SAVE PREDICTION TO DATABASE
            # --------------------------------------------------
            save_prediction(
                date=str(selected_date),
                time=f"{selected_hour:02d}:00",
                predicted=float(prediction),
                actual=actual,
                model_version="RandomForest_v1"
            )

            st.success(
                "Prediction saved to database successfully!"
            )

            # --------------------------------------------------
            # MLFLOW EXPERIMENT TRACKING
            # --------------------------------------------------
            try:

                # Use SQLite backend instead of file-based tracking
                mlflow.set_tracking_uri(
                    f"sqlite:///{BASE_DIR.as_posix()}/mlflow.db"
                )

                mlflow.set_experiment(
                    "Energy_Consumption_Prediction"
                )

                with mlflow.start_run():

                    mlflow.log_param(
                        "model_type",
                        "Random Forest Regression"
                    )

                    mlflow.log_param(
                        "model_version",
                        "RandomForest_v1"
                    )
                    mlflow.log_param(
                        "dataset_version", 
                        "EnergyDataset_v1"
                    )
                    mlflow.log_param(
                        "pipeline_version",
                        "Pipeline_v1"
                    )

                    mlflow.log_param(
                        "prediction_date",
                        str(selected_date)
                    )

                    mlflow.log_param(
                        "prediction_hour",
                        int(selected_hour)
                    )

                    mlflow.log_metric(
                        "predicted_consumption",
                        float(prediction)
                    )

                    if actual is not None:

                        mlflow.log_metric(
                            "actual_consumption",
                            float(actual)
                        )

                        mlflow.log_metric(
                            "absolute_error",
                            float(error)
                        )

                st.success(
                    "MLflow experiment tracking recorded successfully!"
                )

            except Exception as e:

                st.warning(
                    f"Prediction completed, but MLflow tracking "
                    f"could not be recorded: {e}"
                )

            st.divider()

            # --------------------------------------------------
            # ACTUAL VS PREDICTED VISUALIZATION
            # --------------------------------------------------
            st.subheader(
                "📊 Actual vs Predicted Consumption"
            )

            if actual is not None:

                fig, ax = plt.subplots(figsize=(8, 4))

                ax.bar(
                    ["Actual", "Predicted"],
                    [actual, prediction]
                )

                ax.set_ylabel(
                    "Global Active Power (kW)"
                )

                ax.set_title(
                    "Actual vs Predicted Consumption"
                )

                st.pyplot(fig)

                plt.close(fig)

            else:

                st.info(
                    "Actual consumption is unavailable for comparison."
                )

            # --------------------------------------------------
            # FEATURE VALUES USED
            # --------------------------------------------------
            with st.expander("View model input features"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        else:

            st.warning(
                "No data available for the selected time."
            )

else:

    st.warning(
        "No data available for the selected date."
    )

st.divider()

# --------------------------------------------------
# DAILY CONSUMPTION TREND
# --------------------------------------------------
st.header("📈 Consumption Trend Analysis")

daily_data = (
    df.groupby(df["Datetime"].dt.date)["Actual"]
    .mean()
    .reset_index()
)

daily_data.columns = [
    "Date",
    "Average Consumption"
]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    daily_data["Date"],
    daily_data["Average Consumption"]
)

ax.set_xlabel("Date")
ax.set_ylabel(
    "Average Global Active Power (kW)"
)

ax.set_title(
    "Daily Energy Consumption Trend"
)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# --------------------------------------------------
# HISTORICAL DATASET
# --------------------------------------------------
st.header("📋 Historical Dataset")

st.write(
    "Explore the historical energy consumption records "
    "used by the prediction system."
)

st.dataframe(
    df,
    use_container_width=True,
    height=400
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Energy Consumption Prediction System | "
    "Random Forest Regression | Streamlit"
)