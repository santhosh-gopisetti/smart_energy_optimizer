import streamlit as st
import pandas as pd
import pickle

# Load trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Smart Energy Optimizer", layout="wide")

# ======= Custom Header =======
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            margin-bottom: 40px;
            color: #666;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Smart Energy Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI tool to assess energy efficiency in telecom infrastructure</div>', unsafe_allow_html=True)

# ======= Tabs for Live and Batch Mode =======
tab1, tab2 = st.tabs(["Live Prediction", "Batch Upload"])

# =================== TAB 1: Live Prediction ===================
with tab1:
    st.header("Enter Sensor Readings")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 30.0)
        battery_health = st.number_input("Battery Health (%)", 0.0, 100.0, 80.0)
        uptime = st.number_input("Uptime (Hours)", 0, 48, 12)

    with col2:
        voltage = st.number_input("Voltage (V)", 0.0, 300.0, 230.0)
        site_type = st.selectbox("Site Type", ["Ground", "Rooftop"])

    with col3:
        power_usage = st.number_input("Power Usage (W)", 0.0, 5000.0, 1000.0)
        location_type = st.selectbox("Location Type", ["Rural", "Urban"])

    # Encode categorical values
    site_type_rooftop = 1 if site_type == "Rooftop" else 0
    location_type_urban = 1 if location_type == "Urban" else 0

    input_df = pd.DataFrame([{
        "temperature": temperature,
        "Voltage": voltage,
        "power_usage": power_usage,
        "battery_health": battery_health,
        "uptime": uptime,
        "site_type_Rooftop": site_type_rooftop,
        "location_type_Urban": location_type_urban
    }])

    # Prediction
    if st.button("Predict Energy Efficiency"):
        prediction = model.predict(input_df)[0]
        st.subheader(f"Predicted Efficiency: {prediction:.2f}%")
        st.progress(min(int(prediction), 100))

        # Risk classification
        if prediction < 50:
            st.error("Risk: Low Efficiency – Immediate attention needed.")
        elif prediction < 75:
            st.warning("Risk: Moderate Efficiency – Can be improved.")
        else:
            st.success("Risk: High Efficiency – System is optimal.")

        # Diagnostic Suggestions
        st.subheader("Diagnostic Suggestion")
        issues = []
        if battery_health < 60:
            issues.append(f"Low Battery Health: {battery_health}%")
        if power_usage > 2000:
            issues.append(f"High Power Usage: {power_usage}W")
        if temperature > 40:
            issues.append(f"High Temperature: {temperature}°C")
        if voltage < 210:
            issues.append(f"Low Voltage: {voltage}V")

        if issues:
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.write("Efficiency is slightly low, but no major cause detected.")

# =================== TAB 2: Batch Upload ===================
with tab2:
    st.header("Upload CSV File for Batch Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file with sensor readings", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            expected_cols = [
                "temperature", "Voltage", "power_usage", "battery_health", "uptime",
                "site_type_Rooftop", "location_type_Urban"
            ]

            if not all(col in batch_df.columns for col in expected_cols):
                st.error("CSV is missing one or more required columns.")
            else:
                predictions = model.predict(batch_df)
                batch_df["Predicted_Efficiency (%)"] = predictions.round(2)
                st.success("Predictions generated successfully.")
                st.dataframe(batch_df)

                csv_download = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_download,
                    file_name="efficiency_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error: {e}")
