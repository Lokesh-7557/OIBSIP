#importing libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

#loading data
Car_data = pd.read_csv("car data.csv")

print(Car_data.head())
print("Shape of data :", Car_data.shape)
print(Car_data.info())
print(Car_data.describe())

#Checking and Removing null values
print("Null values in dataset :\n", Car_data.isnull().sum())

#Checking and Removing Duplicate values
print("Dupicate values in dataset :", Car_data.duplicated().sum())
Car_data.drop_duplicates(inplace=True)
print("Shape of dataset after removing Dupicate values :", Car_data.shape)

#ANALYZING DATA
for col in Car_data.columns:
    print("Unique values of " +col)
    print(Car_data[col].unique())
    print("==========================\n")

#Defining Feature variable
X = Car_data.drop(columns=["Selling_Price"])
print(X)

#Defining Target variable
y = Car_data["Selling_Price"]
print(y)

#finding Catrgotical Columns
Categorical_cols = X.select_dtypes(include=['object']).columns
print(Categorical_cols)

# Applying OneHotEncoder to categorical columns
ecd = OneHotEncoder(drop='first', sparse_output=False)
encoded_Car_data = ecd.fit_transform(X[Categorical_cols])
print(encoded_Car_data)

encoded_Car_df = pd.DataFrame(encoded_Car_data, columns=ecd.get_feature_names_out(Categorical_cols), index=X.index)
print(encoded_Car_df)

X = pd.concat([X.drop(columns=Categorical_cols), encoded_Car_df], axis=1)
print(X)

# #Splitting Data into Testing and Training Set
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size = 0.3, random_state=40 ) 

#Creating and Training the model
model = LinearRegression()
model.fit(X_train, y_train)

predict = model.predict(X_test)
print(predict)

error_score = metrics.r2_square(y_train, predict)
print("R squared error : ", error_score)
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
# print(f"R-squared: {r2_score(y_test, y_pred)}")



