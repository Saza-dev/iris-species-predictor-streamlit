import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load and cache the iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Load data and target names
df, target_names = load_data()

# Train model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar for input
st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# Format input data
input_data = pd.DataFrame ([[sepal_length, sepal_width, petal_length, petal_width]],columns=df.columns[:-1])

# Predict
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Output result
st.title("Iris Flower Species Prediction")
st.write(f"### The predicted species is: **{predicted_species}**")
