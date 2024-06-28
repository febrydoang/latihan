import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
data = pd.read_csv('data/water_potability.csv')

# Display data
st.write(data.head())

# EDA
fig, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), ax=ax, annot=True)
st.pyplot(fig)

# Preprocessing
data['ph'].fillna(value=data['ph'].median(), inplace=True)
data['Trihalomethanes'].fillna(value=data['Trihalomethanes'].median(), inplace=True)
data = data.dropna()

X = data.drop('Potability', axis=1).values
y = data['Potability'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Modeling
model = Sequential()
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test), verbose=1)

st.write("Model training completed")
