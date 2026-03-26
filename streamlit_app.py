import streamlit as st
import sys
from crop_model import load_model, predict_crop

# -------------------------------
# Streamlit UI
# -------------------------------
def run_app(model_name):
    st.set_page_config(page_title="Crop Recommendation System")
    st.title("🌾 Crop Recommendation System")

    st.markdown("""
    Enter soil and weather parameters to get the recommended crop.
    """)

    N = st.number_input("Nitrogen", 0, 140, value=50)
    P = st.number_input("Phosphorus", 0, 145, value=50)
    K = st.number_input("Potassium", 0, 205, value=50)
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, value=50.0)
    ph = st.number_input("Soil pH", 3.5, 9.5, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, value=100.0)

    # model = load_model("RandomForest")
    model = load_model(model_name.replace(" ",""))

    if st.button("🌱 Recommend Crop"):
        crop = predict_crop(model, [N, P, K, temp, humidity, ph, rainfall])
        st.success(f"Recommended Crop: **{crop.capitalize()}**")

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Random Forest"
    run_app(model_name)
