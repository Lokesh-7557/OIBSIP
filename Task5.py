#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
 

#loading data
data = pd.read_csv("Advertising.csv")

print(data.head())
print("Shape of dataset :", data.shape)
print(data.info())
print(data.describe())

#Checking for null & Duplicate values
print("Null values in dataset :\n",data.isnull().sum())

print("Duplicate values in dataset :", data.duplicated().sum())

#Drop useless columns
data=data.drop(columns=['Unnamed: 0'])
print("Dataset after droping column :\n", data.head())
print("Sahpe of clean dataset :", data.shape)
print("Columns of clean dataset :", data.columns)

#Data Visualization
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales')
plt.show()

#Function to plot histogram
def hist(Device, color):
    data[Device].plot.hist(bins=10, color = color, xlabel=Device)
    plt.show()

hist("TV", "Blue")
hist("Radio", "Green")
hist("Newspaper", "Red")

sns.heatmap(data.corr(), annot=True)
plt.show()

#testing and training the model
X = data.drop('Sales', axis=1)
print(X)
y = data['Sales']
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)

model=LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(prediction)

