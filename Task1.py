#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#loading data
iris_data = pd.read_csv("Iris.csv")


print(iris_data.info())
print(iris_data.shape)
print(iris_data.describe)
print(iris_data.head())

#deleting Id cloumn
iris_data = iris_data.drop(columns=["Id"])

print(iris_data.describe())
print(iris_data.head())

#Finding Species count
print(iris_data["Species"].value_counts())

#Analysing data
sns.pairplot(iris_data, hue="Species")
plt.show()  

X = iris_data.drop("Species", axis=1)
print(X)

y = iris_data["Species"]
print(y)

#Testing & Training Data
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=10)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#Checking Accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

#Testing
new_data = pd.DataFrame({"SepalLengthCm": [2.4], "SepalWidthCm": [5.4], "PetalLengthCm": [3.2], "PetalWidthCm": [1.2]})

prediction = knn.predict(new_data)
print("Species is :" ,prediction[0])