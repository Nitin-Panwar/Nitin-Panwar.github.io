---
layout: post
title: A Gentle Introduction to xgboost machine learning model
published: False
---

__Xgboost__ statnds for eXtreme Gradient Boosting, It is an implementation of __gradient boosted decision tree__ desigend for speed and performance. Let's break it down further, and understand it one by one. 

##### Boosting Algorithm:- 

"The term __Boosting__ refers to a family of algorithms which converts weak learner to strong learners". 
Boosting combines weak learner a.k.a. base learner to form a strong rule.

Boosting is an ensemble technique in which the predictors are not made independently(As in case of bagging), but sequentially.

To find weak rule, we apply base learning (ML) algorithms(Decision tree in case of xgboost) with a different distribution. Each time base learning algorithm is applied, it generates a new weak prediction rule. This is an iterative process. After many iterations, the boosting algorithm combines these weak rules into a single strong prediction rule.

Step 1:  The base learner takes all the distributions and assign equal weight or attention to each observation.

Step 2: If there is any prediction error caused by base learning algorithm, then we pay higher attention to the observations having prediction error. Then, we again apply base learning algorithm.

Step 3: Iterate Step 2 till the limit of base learning algorithm is reached or higher accuracy is achieved.

Finally, it combines the outputs from weak learner and creates  a strong learner which eventually improves the prediction power of the model. Boosting pays higher focus on examples which are mis-classiﬁed or have higher errors by preceding weak rules.

![Imgur](https://i.imgur.com/ms7EQ9h.jpg)


Let's try to see how bagging is different from boosting.

![Imgur](https://i.imgur.com/OSmqyCQ.png)

Now i guess, you must be good with boosing algorithm. Let's move ahead.

#### Gradient Boosting Algorithm:-

Because new predictors are learning from mistakes committed by previous predictors, it takes less time/iterations to reach close to actual predictions. But we have to choose the stopping criteria carefully or it could lead to overfitting on training data. 

There are many optimization methods, if we use gradient descent as optimization algorithm for finding the minimum of a function then this type of boosting algo is called Gradient Boosting Algorithm.  ___Gradient Boosting algo___ is one of the example of boosting algorithm.

#### Gradient boosting decision model or Xgboost:-
If we use decision tree as a base model for ___gradient boosting algorithm___ then we call it as ___Gradient boosting decision tree__. Or in other words, ___Gradient boosting decision tree__ is also called as Xgboost. 

Now let's try to unserstand math behind it- 

The objective of any supervised learning algorithm is to define a loss function and minimize it. Let’s see how math works with Gradient Boosting algorithm. As we saw previously we will be using __Gredient descent algo__ as an optimization method. Now let's say we have mean squared error (MSE) as loss defined as:

![Imgur](https://i.imgur.com/khRcgRM.png)

We want our predictions, such that our loss function (MSE) is minimum. By using __gradient descent algo__ and updating our predictions based on a learning rate, we can find the values where MSE is minimum.

![Imgur](https://i.imgur.com/YDatYfV.png)

So, we are basically updating the predictions such that the sum of our residuals is close to 0 (or minimum) and predicted values are sufficiently close to actual values.










