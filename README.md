# Do (wo)men talk too much in films?  
## Abstract  
In this report, we train logistic regression, LDA, QDA, tree-based, and KNN
models to help us determine the gender of the film’s lead actor. It is done along
with determining the important features that might have helped the classifier in the
predictions. Moreover, we discuss the differences with each models and determine
which model works best for this problem.  
## Introduction  
This report is aimed at measuring whether male or female lead role is predictable from the amount
of dialogue the actors have, the year the film was made, how much money it made and so on. We
have a dataset of 1037 films containing 13 numerical variables characterising information pertinent to
the films. This is a binary classification problem that needs to be implemented to determine male or
female lead actor. The methods implemented are then tuned according to the parameters provided
and finally we evaluate the feature performances.  
## Exploratory Data Analysis  
Analysing the data, we see the distribution of male and female plotting certain characteristics.  

**Do men or women dominate speaking roles in Hollywood movies?**  
We calculated total male actors and total female actors across all films and find that the films
predominantly cast male actors. 31%(3644) actors were females whereas 69%(8070) actors were
male actors in speaking roles in Hollywood movies.
![ttl_male_ttl_female](imgs/ttl_male_ttl_female.png)
<span id="fig:fig_1" label="fig:fig_1"></span>

**Has gender balance in speaking roles changed over time (i.e. years)?**  
The gender balance has largely been tilted towards more male actors landing in speaking roles than
female actors in films released from 1939 to 2015. The ratio remains almost the same all these years.
![male_female_withtime](imgs/male_female_withtime.png)
<span id="fig:fig_2" label="fig:fig_2"></span>

**Do films in which men do more speaking make a lot more money than films in which women speak more?**  
After determining the gender of the actors with most words in each film, we found that the films
which have more dialogues for male actors make more than 7 times the money than the films which
have more dialogues for female actors. From the dataset, it was observed that 12%($14,411) of
marketshare in Revenue goes to films where women spoke the most in comparison to 88%($101,073)
marketshare in Revenue for the films where men spoke the most.  

![spokemore](imgs/spokemore.png)
<span id="fig:fig_3" label="fig:fig_3"></span>

Based on the above observations which favor male lead actors or female, a worst-case classifier would
have no more than 50-55% accuracy to classify the lead actor correctly.  

## Methods of Classification  
We use five main methods of classification: Logistic Regression, LDA, QDA, K-Nearest Neighbour and Tree Based Methods  

### Logistic Regression  
A binary Logistic regression is a binary classifier which takes in the data variables and gives the likelihood in which the data point should lie into one of the two classes. The decision boundary can be either linear or non-linear in nature. Solver used to run logistic regression was 'lbfgs', which comes as the default for this classifier. The max_iter parameter was changed from 100 to 10000 iterations for the best results and all the input variables were considered.  

The training set of 1037 data points are divided in 70/30 test and validation set, giving us a mean accuracy of 85.9% and the following confusion matrix, where 'Male' lead was set as 1 and 'Female' lead as 0:  
|Lead| Female| Male |
|---|------|-----|
|Female |55 | 36 |
|Male | 8 | 213 |

Since retraining the model with 70/30 uses the same data and can cause overfitting, we used K-fold cross validation to hold out some part of training data while retraining the model. In this way we ensured that the model does not run on the same data and overfits.  
As per K-fold cross validation method with 10 "folds" the model was trained on 9 folds and tested on the 10th fold. This was done 10 times and then the performance was averaged out from all the splits, finally we receive an average accuracy of **87.1%** with a standard deviation of 2.9% .  

### Linear Discriminant Analysis (LDA)  
LDA is a linear classifier also used for dimensionality reduction. It works by increasing the variability between the classes but reducing the separability of each data point within the class. This method is popular for preprocessing the data like images to reduce the number of features, but is not so effective when there is non-linearly separable class to deal with. The model assumes that the input data is normally distributed and each class has identical covariance matrices. We ran the LDA using *LinearDiscriminantAnalysis()* method in sklearn library, with default solver ie. *svd* (singular value decomposition)

The training set of 1037 data points are divided in 70/30 train and validation set, giving us a mean accuracy of 84% and the following confusion matrix, where 'Male' lead was set as 1 and 'Female' lead as 0:

|Lead| Female| Male |
|---|------|-----|
|Female |52 | 39 |
|Male | 11 | 210 |

As per K-fold cross validation method with k = 10 and shuffling the data before splitting them in train and validation sets, we receive an average accuracy of **86\%**  with a standard deviation of 2.9%.

### Quadratic Discriminant Analysis (QDA)  

Just like LDA, QDA works on the same principals and is used to classify data that are non-linearly separable. This means that this method works on classes that have nonidentical covariance as opposed to LDA. In a similar fashion, *QuadraticDiscriminantAnalysis()* method is used with the default solver *svd* 

The training set of 1037 data points are divided into 70/30 train and validation set, giving us a mean accuracy of 89.7\% and the following confusion matrix, where 'Male' lead was set as 1 and 'Female' lead as 0:

|Lead| Female| Male |
|---|------|-----|
|Female |65 | 26 |
|Male | 6 | 215 |
    

As per K-fold cross validation method with k = 10 and shuffling the data before splitting them in train and validation sets, we receive an average accuracy of **87.3%** with a standard deviation of 3.8%.


### K-Nearest Neighbour (KNN)  

In a classification task, K-Nearest Neighbor method classifies a given point to a certain categorical label by finding the closest distances between the point and all other K numbers of data points in the dataset. Usually, Euclidean distance is taking as a measure of distance. In practice, to find the best value of K, we use a for loop to calculate the error rates toward different values of K. And then choose the K with lowest error to train the model. An optimal value of K has to be found out because lower value may lead to over-fitting issue, and higher value of K may require more computational complication in distance.

Before building the for loop, we split the dataset into training dataset and validation dataset which is including 30 percent of data. And then do the scaling on the dataset. Because the statistical principle of KNN is to calculate the distances between data points, it is easier to calculate it with the process of normalization. After that, we use a for loop to get the error rates with various values of K from 1 to 50 so that We find the best choice of K is 4 with minimum error rate **20.19%**. Then we train the KNN model using the best K, and compare the prediction results with the validation dataset to have the final accuracy score **79.81%**. The confusion matrix is shown as follows.

|Lead| Female| Male |
|---|------|-----|
|Female |42 | 46 |
|Male | 17 | 207 |


There are some pros of using the KNN algorithm. First, it is easy to implement and runs faster than other algorithms such as linear regression, SVM and so on. Second, there is only one parameter required to be found the best one in KNN, that is, the value of K. There are also some cons of using it. First, it cannot deal with datasets with large amount of dimensions well because it is difficult to calculate Euclidean distance in each dimension. Second, it does not work well with categorical features due to the difficulties in calculating distances with categorical features.


### Tree-Based Methods

For tree-based methods, we use Random Forest and Gradient Boosting. Random Forest uses bags of trees to average out the decision for classification and extreme gradient boosting uses regularization for building trees and classifying. Extreme Gradient Boosting uses weak learners and iterates from previous trees to provide a better prediction minimizing the objective function with each iteration and using space efficient processing. In our project, we find the optimal tree depth to be 7 for random forest, with cross-validation k=10-folds. For Extreme Gradient Boosting, we find the optimal parameters through a grid search with the optimal tree depth of 7 using AUC as the scoring determinant of best score for optimal parameters. For Random Forest we obtain a 90% training accuracy and testing accuracy of **80%**. Whereas for Gradient Boosting, we obtain a 98% training accuracy and **85%** in testing accuracy. 

## Feature Importance

### Playing with features

### Discussion  

## Production method  

## Conclusions  

## References  
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
https://scikit-learn.org/stable/modules/cross_validation.html  
https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html  
https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html  
https://scikit-learn.org/stable/modules/permutation_importance.html  
https://xgboost.readthedocs.io/en/stable/tutorials/model.html  
