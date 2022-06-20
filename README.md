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

### Linear Discriminant Analysis (LDA)  

### Quadratic Discriminant Analysis (QDA)  

### K-Nearest Neighbour (KNN)  

### Tree-Based Methods

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
