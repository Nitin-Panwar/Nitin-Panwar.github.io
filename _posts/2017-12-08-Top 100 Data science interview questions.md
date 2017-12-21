---
layout: post
title: Top 100 Data science interview questions
published: true
---
Data science, also known as data-driven decision, is an interdisciplinery field about scientific methods, process and systems to extract knowledge from data in various forms, and take descision based on this knowledge. A data scientist should not only be evaluated only on his/her knowledge on machine learning, but he/she should also have good expertise on statistics. 
I will try to start from very basics of data science and then slowly move to expert level. So let's get started.

### 1.  What is the difference between supervised and unsupervised machine learning?
#### Supervised Machine learning:
Supervised machine learning requires training labeled data. 
#### Unsupervised Machine learning: 
Unsupervised machine learning doesn't required labeled data. 

### 2. What is bias, variance trade off ?
#### Bias:  
"Bias is error introduced in your model due to over simplification of machine learning algorithm." 
It can lead to underfitting. When you train your model at that time model makes simplified assumptions to make the target function easier to understand.
##### Low bias machine learning algorithms - Decision Trees, k-NN and SVM 
##### Hight bias machine learning algorithms - Liear Regression, Logistic Regression

#### Variance: 
"Variance is error introduced in your model due to complex machine learning algorithm, your model learns noise also from the training dataset and performs bad on test dataset."
It can lead high sensitivity and overfitting.

Normally, as you increase the complexity of your model, you will see a reduction in error due to lower bias in the model. However, this only happens till a particular point. As you continue to make your model more complex, you end up over-fitting your model and hence your model will start suffering from high variance.

![Bias variance trade off](https://i.imgur.com/a7UKJO8.png)

#### Bias, Variance trade off: 
The goal of any supervised machine learning algorithm is to have low bias and low variance to achive good prediction performance.
1. The k-nearest neighbors algorithm has low bias and high variance, but the trade-off can be changed by increasing the value of k which increases the number of neighbors that contribute to the prediction and in turn increases the bias of the model.

2. The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.

There is no escaping the relationship between bias and variance in machine learning.

Increasing the bias will decrease the variance.
Increasing the variance will decrease the bias.


### 3. What is exploding gradients ?
"Exploding gradients are a problem where __large error gradients__ accumulate and result in very large updates to neural network model weights during training."
At an extreme, the values of weights can become so large as to overflow and result in NaN values.

This has the effect of your model being unstable and unable to learn from your training data. 
Now let's understand what is the gradient.
#### Gradient: 
Gradient is the __direction and magnitude__ calculated during training of a neural network that is used to update the network weights in the right direction and by the right amount. 

### 4. What is a confusion matrix ?
The confusion matrix is a 2X2 table that contains 4 outputs provided by the __binary classifier__.  Various measures, such as error-rate, accuracy, specificity, sensitivity, precision and recall are derived from it.
_Confusion Matrix_

![](https://i.imgur.com/NYozwa5.png)

A dataset used for performance evaluation is called test dataset. It should contain the correct labels and predicted labels.

![observed-labels.png](https://i.imgur.com/64mZwRG.png)

The predicted labels will exactly the same if the performance of a binary classfier is perfect.

![perfect-classifier.png](https://i.imgur.com/HeVTzab.png)

The predicted labels usually match with part of the observed labels in real world scenarios. 

![regular-classifier.png](https://i.imgur.com/hqNvN2N.png)
 
 A binary classifier predicts all data instances of a test dataset as either positive or negative. This produces four outcomes-
 1. True positive(TP) - Correct positive prediction
 2. False positive(FP) - Incorrect positive prediction
 3. True negative(TN) - Correct negative prediction
 4. False negative(FN)  - Incorrect negative prediction 
 
 ![perfect-classifier.png](https://i.imgur.com/WxQLV83.png)
 
 __Basic measures derived from the confusion matrix__
 1. Error Rate = (FP+FN)/(P+N)
 2. Accuracy = (TP+TN)/(P+N)
 3. Sensitivity(Recall or True positive rate) = TP/P
 4. Specificity(True negative rate) = TN/N
 5. Precision(Positive predicted value) = TP/(TP+FP)
 6. F-Score(Harmonic mean of precision and recall) = (1+b)(PREC.REC)/(b^2PREC+REC) where b is commonly 0.5, 1, 2.
 
 
### 6. What is selection Bias ?
Selection bias occurs when sample obtained is not represantative of the population intended to be analyzed.
 
### 7. Explain SVM machine learning algorithm in detail.
SVM stands for support vector machine, it is a supervised machine learning algorithm which can be used for both __Regression and Classification__. If you have n features in your training dataset, SVM tries to plot it in n-dimentional space with the value of each feature being the value of a particular coordinate. SVM uses hyper planes to seperate out different classes based on the provided kernel function. 

![regular-classifier.png](https://i.imgur.com/dU1GDSe.png)

### 8. What are the different kernels functions in SVM ? 
There are four types of kernels in SVM.
 1. Linear Kernel
 2. Polynomial kernel
 3. Radial basis kernel
 4. Sigmoid kernel
 
### 9. Explain Decision Tree algorithm in detail.
Decision tree is a supervised machine learning algorithm mainly used for the __Regression and Classification__.It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. Decision tree can handle both categorical and numerical data. 
 
![Imgur](https://i.imgur.com/Y4jdwtw.png)

### 10. What is Enteropy and Information gain in Decision tree algorithm ?
The core algorithm for building decision tree is called __ID3__. __ID3__ uses __Enteropy__ and __Information Gain__ to construct a decision tree. 

__Enteropy__

A decision tree is built top-down from a root node and involve partitioning of data into homogenious subsets. __ID3__ uses enteropy to check the homogeneity of a sample. If the sample is completely homogenious then entropy is zero and if the sample is an equally divided it has entropy of one. 

![Imgur](https://i.imgur.com/XFE0f6V.png)


__Information Gain__ 

The __Information Gain__ is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about finding attributes that returns the highest information gain.

![Imgur](https://i.imgur.com/KI934dk.png)

![Imgur](https://i.imgur.com/rZcG68J.png)


### 11. What is pruning in Decision Tree ?
When we remove sub-nodes of a decision node, this procsss is called pruning or opposite process of splitting. 


### 12. What is Ensemble Learning ?
Ensemble is the art of combining diverse set of learners(Individual models) togather to improvise on the stability and predictive power of the model. Ensemble learning has many types but two more popular ensemble learning techniques are mentioned below.

__Bagging__

Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions. In generalized bagging, you can use different learners on different population. As you expect this helps us to reduce the variance error.

![Imgur](https://i.imgur.com/1pE4VTo.png)

__Boosting__

Boosting is an iterative technique which adjust the weight of an observation based on the last classification. If an observation was classfied incorrectly, it tries to increase the weight of this observation and vice versa. Boosting in general decreases the bias error and builds strong predictive models. However, they may overfit on the training data. 

![Imgur](https://i.imgur.com/mH85e7T.png)

### 13. What is Random Forest? How does it work ?
Random forest is a versatile machine learning method capable of performing both regression and classification tasks. It is also used for dimentionality reduction, treats missing values, outlier values. It is a type of ensemble learning method, where a group of weak models combine to form a powerful model. 

In Random Forest, we grow multiple trees as opposed to a single tree. To classify a new object based on attributes, each tree gives a classification. The forest chooses the classification having the __most votes__(Over all the trees in the forest) and in case of regression, it takes the __average__ of outputs by different trees. 




 
 # To Be Continued... 
 
 
 
 
 

 




  

