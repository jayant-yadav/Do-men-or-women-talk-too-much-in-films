import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import sklearn.preprocessing as skl_pre
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight

url = 'train.csv'
movies_df = pd.read_csv(url)
print(movies_df.dtypes)

print(movies_df.isnull().sum()) #checking for null values

movies_df['Number of words lead'] = movies_df['Number of words lead'].astype(int)
movies_df['Age Lead'] = movies_df['Age Lead'].astype(int)
movies_df['Age Co-Lead'] = movies_df['Age Co-Lead'].astype(int)
movies_df['Lead'].replace('Male',1,inplace=True)
movies_df['Lead'].replace('Female',0,inplace=True)
print(movies_df.dtypes)

movies_df.describe()

pd.plotting.scatter_matrix(movies_df.iloc[:,:],figsize=(30,30))
plt.show()

plt.figure(figsize=(13,10))
sns.heatmap(movies_df.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
feature_df = movies_df.drop(["Lead"], axis=1)
label_df = movies_df["Lead"]
cv = KFold(n_splits=10, random_state=1, shuffle=True)

"""Logistic Regression"""

logreg_model = skl_lm.LogisticRegression(max_iter = 10000)

logreg_model.fit(train_x,train_y)
pred_y=logreg_model.predict(test_x)
matrix = metrics.confusion_matrix(test_y, pred_y)
print(matrix)
print(f"Accuracy: {np.mean(pred_y == test_y):.3f}")

scores = cross_val_score(logreg_model, feature_df, label_df, scoring='accuracy',
        cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
