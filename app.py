import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🌸 Iris Flower Classification")

# Load dataset
df = pd.read_csv("iris.csv")

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Sliders under the title
st.subheader("Enter Flower Measurements")
sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.6)
sepal_width  = st.slider("Sepal width", 2.0, 4.5, 3.79)
petal_length = st.slider("Petal length", 1.0, 7.0, 1.80)
petal_width  = st.slider("Petal width", 0.1, 2.5, 1.65)

# Submit button
if st.button("🔍 Submit"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
# Show result at the bottom
st.subheader("Predicted Species")
st.success(f"🌼 {prediction}")


