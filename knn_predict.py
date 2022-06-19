# import the libraries
import numpy as np
import matplotlib
# this setting is due to my computer cannot output a plot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# import the dataset 
# I have downloaded to my personal computer from the website
dataset = pd.read_csv('C:/Users/Danlonese/Desktop/SML group project/train.csv')
# show the 5 rows of the dataframe
dataset.head()
# show the information about the dataset
dataset.info(verbose=True)
# there are 1039 examples and 14 features without null
# "Lead" is object type, others are numerical features

###############  Task 2 of KNN  ####################

# separate the variables into dependent variable and independent variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 13].values

# split the dataset into the training dataset and validation dataset
X_train, X_vali, Y_train, Y_vali = train_test_split(X, Y, test_size=0.3, 
                                    random_state= 22)

# data is scaled to mean 0 and variance 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_vali = sc.transform(X_vali)

# calculate error to find the best value of K
error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pre_i = knn.predict(X_vali)
    error.append(np.mean(pre_i != Y_vali))

# method 1: directly print the minimum error rate and corresponding value of K
print(min(error))
print(error.index(min(error))) # the result is to find the index of min error, 
#so we should plus one on it to get the K
# From the result, the minimum error rate is 0.20192 and the best K value is 4

# method 2: plot the error rate toward different values of K
plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='blue', marker='o', markerfacecolor='yellow',
    markersize=10)
plt.title('Error Rate of different values of K')
plt.xlabel('The value of K')
plt.ylabel('Error Rate')
# The result of method 2 is the same to the one in method 1

# training the KNN model with the best K=4 on the training dataset
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, Y_train)
# predict by using the validation dataset
Y_pred = knn.predict(X_vali)

# evaluate performance of the model through confusion matrix, classification report 
# and accuracy score
cm = confusion_matrix(Y_vali, Y_pred)
print(cm)
cr = classification_report(Y_vali, Y_pred)
print(cr)
acs = accuracy_score(Y_vali, Y_pred)
print(acs)
# the accuracy score is 79.81%

#################### Task 3 of KNN #######################
# delete the features "Number words female"[0] and "Number words male"[7]
X1 = dataset.iloc[:, [1,2,3,4,5,6,8,9,10,11,12]].values
Y1 = dataset.iloc[:, 13].values

# repeat the process in task 2
X_train, X_vali, Y_train, Y_vali = train_test_split(X1, Y1, test_size=0.3, 
random_state= 22)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_vali = sc.transform(X_vali)
error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pre_i = knn.predict(X_vali)
    error.append(np.mean(pre_i != Y_vali))
print(min(error))
print(error.index(min(error)))
# The error rate is 0.22115 and the best value of K is 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_vali)
acs = accuracy_score(Y_vali, Y_pred)
print(acs)
# the accuracy score is 77.88% lower than 79.81%
# Therefore, words spoken by males and females are important features

# delete the features "Year"[5]
X2 = dataset.iloc[:, [0,1,2,3,4,6,7,8,9,10,11,12]].values
Y2 = dataset.iloc[:, 13].values

# repeat the process in task 2
X_train, X_vali, Y_train, Y_vali = train_test_split(X2, Y2, test_size=0.3,
    random_state= 22)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_vali = sc.transform(X_vali)
error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pre_i = knn.predict(X_vali)
    error.append(np.mean(pre_i != Y_vali))
print(min(error))
print(error.index(min(error)))
# The error rate is 0.20833 and the best value of K is 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_vali)
acs = accuracy_score(Y_vali, Y_pred)
print(acs)
# the accuracy score is 79.17% which is little lower than 79.81%
# Therefore, Year of release is not an important feature

# delete the features "Gross"[8]
X3 = dataset.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12]].values
Y3 = dataset.iloc[:, 13].values

# repeat the process in task 2
X_train, X_vali, Y_train, Y_vali = train_test_split(X3, Y3, test_size=0.3,
    random_state= 22)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_vali = sc.transform(X_vali)
error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pre_i = knn.predict(X_vali)
    error.append(np.mean(pre_i != Y_vali))
print(min(error))
print(error.index(min(error)))
# The error rate is 0.20833 and the best value of K is 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_vali)
acs = accuracy_score(Y_vali, Y_pred)
print(acs)
# the accuracy score is 79.17% which is little lower than 79.81%
# Therefore, money made by film is not an important feature