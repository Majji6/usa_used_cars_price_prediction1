import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("usa_cars.pkl")

# Load dataset (used only for dropdown values)
df = pd.read_csv("USA_cars_datasets.csv")

st.title("🚗 USA Car Price Prediction")

# ---------- Input Fields ----------

brand = st.selectbox(
    "Brand",
    sorted(df["brand"].dropna().unique())
)

model_name = st.selectbox(
    "Model",
    sorted(df["model"].dropna().unique())
)

year = st.selectbox(
    "Year",
    sorted(df["year"].unique())
)

title_status = st.selectbox(
    "Title Status",
    sorted(df["title_status"].dropna().unique())
)

mileage = st.number_input(
    "Mileage",
    min_value=float(df["mileage"].min()),
    max_value=float(df["mileage"].max()),
    value=float(df["mileage"].median())
)

color = st.selectbox(
    "Color",
    sorted(df["color"].dropna().unique())
)

vin = st.selectbox(
    "VIN",
    sorted(df["vin"].dropna().unique())
)

lot = st.selectbox(
    "Lot",
    sorted(df["lot"].unique())
)

state = st.selectbox(
    "State",
    sorted(df["state"].dropna().unique())
)

country = st.selectbox(
    "Country",
    sorted(df["country"].dropna().unique())
)

condition = st.selectbox(
    "Condition",
    sorted(df["condition"].dropna().unique())
)

# ---------- Prediction ----------

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "brand": [brand],
        "model": [model_name],
        "year": [year],
        "title_status": [title_status],
        "mileage": [mileage],
        "color": [color],
        "vin": [vin],
        "lot": [lot],
        "state": [state],
        "country": [country],
        "condition": [condition]
    })

    prediction = model.predict(input_df)

    st.success(f"Predicted Price: ${prediction[0]:,.2f}")