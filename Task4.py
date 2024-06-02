#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading Data
spam_data = pd.read_csv("spam.csv", encoding='latin-1')

print(spam_data.head())
print("Information about the dataset :\n", spam_data.info)
print("Shape of the dataset :", spam_data.shape)
print(spam_data.describe())
print("Columns of dataset :", spam_data.columns)

#Checking and removing null values
print(spam_data.isnull().sum())
spam_data.drop(columns={'Unnamed: 2',    'Unnamed: 3',    'Unnamed: 4'}, inplace=True)
print(spam_data.head())
print("Shape of data after cleaning :", spam_data.shape)
print(spam_data.isnull().sum())

#Renaming remaining columns
spam_data=spam_data.rename(columns={'v1' : 'Category', 'v2' : 'Message'})
print(spam_data.head())

#Lableing the columns
spam_data.loc[spam_data['Category'] == 'spam','Category',]=0
spam_data.loc[spam_data['Category'] == 'ham','Category',]=1

X = spam_data['Message']
y = spam_data['Category']

print(X)
print(y)

#splitting data into testing and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=3)


feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

#Creating and training model
model = LogisticRegression()
model.fit(X_train_features, y_train)

#Making predictons
prediction_on_test_data = model.predict(X_test_features)
print("Accuracy score :", accuracy_score(y_test, prediction_on_test_data))

print(spam_data.head(1))
input = ["Ela kano.,il download, come wen ur free.."]
input_mail = feature_extraction.transform(input)

prediction = model.predict(input_mail)
print(prediction)

if prediction[0] == 1:
    print("ham mail")
else:
    print("spam mail")