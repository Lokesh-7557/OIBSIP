#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading Data
UE_data = pd.read_csv("Unemployment in India.csv")

print(UE_data.head())
print("Shape of Data is : ", UE_data.shape)
print(UE_data.info())
print(UE_data.describe())

UE_data = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

print(UE_data.head())
print("Shape of Data is : ", UE_data.shape)
print(UE_data.info())
print(UE_data.describe())

#Checking null values in dataset upto 11_2020
print("Null values in Dataset upto 11_2020 :\n", UE_data.isnull().sum())

UE_data[' Date'] = pd.to_datetime(UE_data[' Date'])

#Analyzing Data

plt.figure(figsize=(15, 5))
plt.bar(UE_data[" Date"], UE_data[" Estimated Unemployment Rate (%)"])
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Uemployment Rate in India in India During Covid-19")
plt.show()


plt.figure(figsize=(15,5))
sns.lineplot(x = " Date", y = " Estimated Unemployment Rate (%)", data=UE_data)
plt.title("Unemployment Rate in India During Covid-19")
plt.show()


plt.figure(figsize=(15, 5))
sns.histplot(UE_data[' Estimated Unemployment Rate (%)'], bins=20, kde=True, color='blue', alpha=0.6)
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Unemployment Rate During Covid-19")
plt.grid(True)
plt.show()


mean_unemployment = UE_data[' Estimated Unemployment Rate (%)'].mean()
median_unemployment = UE_data[' Estimated Unemployment Rate (%)'].median()
print('Mean Unemployment Rate during Covid-19:', {mean_unemployment})
print('Median Unemployment Rate during Covid-19: ',{median_unemployment})
