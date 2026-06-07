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

# Sidebar inputs
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal length", float(X["sepal_length"].min()), float(X["sepal_length"].max()))
sepal_width  = st.sidebar.slider("Sepal width",  float(X["sepal_width"].min()),  float(X["sepal_width"].max()))
petal_length = st.sidebar.slider("Petal length", float(X["petal_length"].min()), float(X["petal_length"].max()))
petal_width  = st.sidebar.slider("Petal width",  float(X["petal_width"].min()),  float(X["petal_width"].max()))

# Prediction
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write("### 🌼 Predicted Species:", prediction[0])

# Show dataset preview
st.subheader("📊 Dataset Snapshot")
st.write(df.head())
