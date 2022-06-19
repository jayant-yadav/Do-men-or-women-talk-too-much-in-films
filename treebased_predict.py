#!/usr/bin/env python
# coding: utf-8

# In[7]:
import xgboost as xgb

# In[2]:
import eli5

# In[3]:
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

# In[4]:
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

# In[38]:
from eli5.sklearn import PermutationImportance
from eli5.sklearn import permutation_importance
from eli5.permutation_importance import get_score_importances

# In[45]:
from xgboost import XGBClassifier

# In[47]:
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

# In[48]:
perm = PermutationImportance(model_xgb).fit(X_train,y_train)

# In[39]:
perm_imp_xgb = permutation_importance(bst, X_train,y_train)

# In[41]:
perm_xgb = PermutationImportance(bst).fit(X_train,y_train)

# In[38]:
list(X_test.columns)

# In[49]:
eli5.show_weights(perm, feature_names = list(X_test.columns))

# In[39]:
eli5.show_weights(perm, feature_names = list(X_test.columns))

# In[42]:
f, ax = plt.subplots(1,1, figsize=(10, 5))
sns.scatterplot(x='Number of female actors', y='Number words female', 
    hue="Lead", data=df_tr, ax=ax);

# In[41]:
f, ax = plt.subplots(1,1, figsize=(10, 5))
sns.scatterplot(x='Number of female actors', y='Number of male actors', 
    hue="Lead", data=df_tr, ax=ax);

# In[ ]:
f, ax = plt.subplots(1,1, figsize=(10, 5))
sns.scatterplot(x='Number of female actors', y='Number words female', 
    hue="Lead", data=df_tst, ax=ax);

# In[6]:
df_tr = pd.read_csv("train.csv")
df_tst = pd.read_csv("test.csv")
df_tr.head()

# In[15]:
c=['Number words female','Year','Gross']
df_tr.drop(c, axis=1)

# In[16]:
df_tr.head()

# In[9]:
df_tr[df_tr.columns[:-1]]

# In[17]:
y_tr = df_tr["Lead"]
x_tr = df_tr[df_tr.columns[:-1]]

# In[15]:
sns.histplot(data=df_tr, x="Lead")

# In[12]:
ggplot(data = df_tr) +
  geom_bar(mapping = aes(x = Lead))

# In[11]:
df_tr

# In[18]:
mean_CV_acc = {}
all_CV_acc = {}
tree_depth_start, tree_depth_end, steps = 3, 31, 4
for i in range(tree_depth_start, tree_depth_end, steps):
    model = DecisionTreeClassifier(max_depth=i)
    score = cross_val_score(estimator=model, X=x_tr, y=y_tr, cv=10,
        n_jobs=-1)
    all_CV_acc[i] = score
    mean_CV_acc[i] = score.mean()

# In[19]:
best_depth = sorted(mean_CV_acc, key=mean_CV_acc.get, reverse=True)[0]
print("The best depth was found to be:", best_depth)

# In[52]:
#remove features of interest
x_tr_f = x_tr.drop(c,axis=1)

# In[53]:
#Test and train for train ds
X_train, X_test, y_train, y_test = train_test_split(x_tr_f, y_tr, 
                            test_size=0.33, random_state=42)

# In[21]:
#Evalaute the performance at the best depth for tree classifier
model_tree = DecisionTreeClassifier(max_depth=best_depth)
model_tree.fit(X_train, y_train)

#Check Accuracy of Train and Test Set
acc_trees_training = accuracy_score(y_train, model_tree.predict(X_train))
acc_trees_testing  = accuracy_score(y_test,  model_tree.predict(X_test))

print("Simple Decision Trees: Accuracy, 
        Training Set \t : {:.2%}".format(acc_trees_training))
print("Simple Decision Trees: Accuracy, 
        Testing Set \t : {:.2%}".format(acc_trees_testing))

# In[22]:
#Fit a Random Forest Model

new_depth = best_depth + 20 

#Training
model = RandomForestClassifier(n_estimators=int(X_train.shape[1]/2), 
            max_depth=new_depth)
model.fit(X_train, y_train)

#Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#Perfromance Evaluation
acc_random_forest_deeper_training = accuracy_score(y_train, y_pred_train)*100
acc_random_forest_deeper_testing = accuracy_score(y_test, y_pred_test)*100

print("Random Forest: Accuracy, 
        Training Set (Deeper): {:0.2f}%".format(acc_random_forest_deeper_training))
print("Random Forest: Accuracy, 
        Testing Set (Deeper):  {:0.2f}%".format(acc_random_forest_deeper_testing))


# In[23]:
print("Training Accuracies:")
print("Decision Trees:\tAccuracy, 
        Training Set \t: {:.2%}".format(acc_trees_training))
print("Bagging: \tAccuracy, 
        Training Set \t: {:0.2f}%".format(acc_bagging_training))
print("Random Forest: \tAccuracy, 
        Training Set \t: {:0.2f}%".format(acc_random_forest_training))
print("RF Deeper: \tAccuracy, 
        Training Set \t: {:0.2f}%".format(acc_random_forest_deeper_training))

# In[24]:
from itertools import product

# In[25]:
from itertools import product
from collections import OrderedDict
param_dict = OrderedDict(
    n_estimators = [400, 600, 800],
    max_features = [0.2, 0.4, 0.6, 0.8]
)


# In[26]:
results = {}
estimators= {}
for ntrees, maxf in product(*param_dict.values()):
    params = (ntrees, maxf)
    est = RandomForestClassifier(oob_score=True, 
                                n_estimators=ntrees, 
                                max_features=maxf, max_depth=50, n_jobs=-1)
    est.fit(X_train, y_train)
    results[params] = est.oob_score_
    estimators[params] = est
outparams = max(results, key = results.get)
outparams


# In[27]:
rf1 = estimators[outparams]


# In[28]:
results

# In[24]:
rf1.score(X_test, y_test)


# In[25]:
#Feature Importance (Gini) for RF
pd.Series(rf1.feature_importances_,index=
        list(X_train)).sort_values().plot(kind="barh")


# In[14]:
import seaborn as sns


# In[27]:
f, ax = plt.subplots(1,1, figsize=(10, 5))
sns.scatterplot(x="Number words female", y="Age Lead", 
            hue="Lead", data=df_tr, ax=ax);


# In[55]:
#label encoding for XGBoost
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


# In[60]:
from sklearn.model_selection import GridSearchCV


# In[70]:
gsearch2.cv_results_


# In[76]:
gsearch2.best_params_


# In[75]:
param_test2 = {'max_depth':range(7,31,2), 'min_samples_split':range(10,50,15)}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, 
            n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4, cv=5)
gsearch2.fit(X_train,y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


# In[31]:
import time
# Create the training and test data
dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
param = {
    'max_depth': best_depth,  # the maximum depth of each tree
    'eta': 0.3,               # the training step for each iteration
    'silent': 1,              # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 2}           # the number of classes that exist in this datset

# Number of training iterations
num_round = 200  

# Start timer
start = time.time()

# Train XGBoost
bst = xgb.train(param, 
                dtest, 
                num_round, 
                evals= [(dtrain, 'train')], 
                early_stopping_rounds=20, # early stopping
                verbose_eval=20)


# Make prediction training set
preds_train = bst.predict(dtrain)
best_preds_train = np.asarray([np.argmax(line) for line in preds_train])

# Make prediction test set
preds_test = bst.predict(dtest)
best_preds_test = np.asarray([np.argmax(line) for line in preds_test])

# Performance Evaluation 
acc_XGBoost_training = accuracy_score(y_train, best_preds_train)*100
acc_XGBoost_test = accuracy_score(y_test, best_preds_test)*100

# Stop Timer
end = time.time()
elapsed_xgboost = end - start

print("XGBoost:\tAccuracy, Training Set \t: {:0.2f}%".format(acc_XGBoost_training))
print("XGBoost:\tAccuracy, Testing Set \t: {:0.2f}%".format(acc_XGBoost_test))


# In[21]:
x_t = xgb.DMatrix(df_tst)
fy = bst.predict(x_t)
fy_test = np.asarray([np.argmax(line) for line in preds_test])


# In[22]:
fy_test


# In[56]:
#XGBoost with features from c removed
import time
# Create the training and test data
dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
param = {
    'max_depth': best_depth,  # the maximum depth of each tree
    'eta': 0.3,               # the training step for each iteration
    'silent': 1,              # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 2}           # the number of classes that exist in this datset

# Number of training iterations
num_round = 200  

# Start timer
start = time.time()

# Train XGBoost
bst = xgb.train(param, 
                dtest, 
                num_round, 
                evals= [(dtrain, 'train')], 
                early_stopping_rounds=20, # early stopping
                verbose_eval=20)


# Make prediction training set
preds_train = bst.predict(dtrain)
best_preds_train = np.asarray([np.argmax(line) for line in preds_train])

# Make prediction test set
preds_test = bst.predict(dtest)
best_preds_test = np.asarray([np.argmax(line) for line in preds_test])

# Performance Evaluation 
acc_XGBoost_training = accuracy_score(y_train, best_preds_train)*100
acc_XGBoost_test = accuracy_score(y_test, best_preds_test)*100

# Stop Timer
end = time.time()
elapsed_xgboost = end - start

print("XGBoost:\tAccuracy, Training Set \t: {:0.2f}%".format(acc_XGBoost_training))
print("XGBoost:\tAccuracy, Testing Set \t: {:0.2f}%".format(acc_XGBoost_test))


# In[ ]:
np.savetxt("predictions.csv", a, delimiter=",")