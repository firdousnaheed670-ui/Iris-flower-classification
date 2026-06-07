import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🌸 Iris Flower Classification")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal length", float(X[:,0].min()), float(X[:,0].max()))
sepal_width  = st.sidebar.slider("Sepal width",  float(X[:,1].min()), float(X[:,1].max()))
petal_length = st.sidebar.slider("Petal length", float(X[:,2].min()), float(X[:,2].max()))
petal_width  = st.sidebar.slider("Petal width",  float(X[:,3].min()), float(X[:,3].max()))

# Prediction
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write("### 🌼 Predicted Species:", iris.target_names[prediction][0])
