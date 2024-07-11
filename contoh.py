# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('water_potability.csv')

# Data preprocessing
data.isnull().sum()
fig, ax = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(), ax = ax, annot = True)
fig, ax = plt.subplots(figsize=(8,8))
abs(data.corr().round(2)['Potability']).sort_values()[:-1].plot.barh(color='c')
data[data['Potability']==0][['ph','Sulfate','Trihalomethanes']].median()
data[data['Potability']==1][['ph','Sulfate','Trihalomethanes']].median()

# Fill missing values
data['ph'].fillna(value=data['ph'].median(), inplace=True)
data['Trihalomethanes'].fillna(value=data['Trihalomethanes'].median(), inplace=True)

# Drop remaining rows with missing values
data = data.dropna()
data.isnull().sum()
data.shape
data.info()

fig, ax = plt.subplots(figsize=(8,8))
abs(data.corr().round(2)['Potability']).sort_values()[:-1].plot.barh(color='c')
data.corr()['Potability'][:-1].sort_values().plot(kind='bar')

trace = go.Pie(labels = ['Potable', 'Not Potable'], values = data['Potability'].value_counts(),
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightskyblue','gold'],
                           line=dict(color='#000000', width=1.5)))

layout = dict(title =  'Distribution of Drinkable Water')
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
plt.figure(figsize = (15,10), tight_layout = True)

for i, feature in enumerate(data.columns):
    if feature != 'Potability':
        plt.subplot(3,3,i+1)
        sns.histplot(data = data, x = feature, palette = 'mako', hue = 'Potability', alpha = 0.5, element="step", hue_order=[1,0])
sns.pairplot(data = data, hue = 'Potability', palette='mako_r', corner=True)

# Data splitting
X = data.drop('Potability', axis=1).values
y = data['Potability'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Data scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('Training shape : ', X_train.shape)
print('Testing shape : ', X_test.shape)

# Model building
model = Sequential()
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

# Model training
model.fit(x=X_train, y=y_train, epochs=300, validation_data=(X_test, y_test), verbose=1)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# Model evaluation
y_pred = model.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")

# Further training with different architectures (optional, for demonstration purposes)
model = Sequential()
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, epochs=900, validation_data=(X_test, y_test), verbose=1)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
y_pred = model.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")

# Another architecture (optional, for demonstration purposes)
model = Sequential()
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=6, activation='tanh'))
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, epochs=500, validation_data=(X_test, y_test), verbose=1)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
y_pred = model.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")

# Streamlit Deployment
import streamlit as st

def main():
    st.title("Water Potability Prediction")
    
    ph = st.number_input("pH")
    hardness = st.number_input("Hardness")
    solids = st.number_input("Solids (TDS)")
    chloramines = st.number_input("Chloramines")
    sulfate = st.number_input("Sulfate")
    conductivity = st.number_input("Conductivity")
    organic_carbon = st.number_input("Organic Carbon")
    trihalomethanes = st.number_input("Trihalomethanes")
    turbidity = st.number_input("Turbidity")
    
    if st.button("Predict"):
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        result = "Potable" if prediction >= 0.5 else "Not Potable"
        st.write(f"The water is {result}")

if __name__ == '__main__':
    main()
