---
layout: post
title: Text Classification using machine learning
published: true
---

___Supervised Text Classification__ The goal is to improve the category classification performance for a set of text posts. The evaluation metric is the macro F1 score.

__Micro F1 Score:__ 
In Micro-average method, you sum up the individual true positives, false positives, and false negatives of the system for different sets and the apply them to get the statistics. 

__Macro F1 Score:__
The method is straight forward. Just take the average of the precision and recall of the system on different sets.


```python
from termcolor import colored
import glob
import pandas as pd
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from time import time
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from nltk import stem
from nltk.corpus import stopwords
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt
%matplotlib inline
```


```python
allfiles = glob.glob(path + '/*.csv')
dfs = [pd.read_csv(filename) for filename in allfiles]
big_df = pd.concat(dfs, ignore_index=True).fillna('None')
```


```python
train_size = int(len(big_df) * 0.8)
print ("Train Size: %d" % train_size)
print ("Test Size: %d" % (len(big_df) - train_size))
train_posts = big_df['text'][:train_size]
train_tags = big_df['label'][:train_size]

test_posts = big_df['text'][train_size:]
test_tags = big_df['label'][train_size:]
```

    Train Size: 33381
    Test Size: 8346



```python
# Compare wordclouds for a couple of categories
jd = " ".join([post for (post,label) in zip(train_posts,train_tags) if label==1])
not_jd = " ".join([post for (post,label) in zip(train_posts,train_tags) if label==0])

jd_cloud = WordCloud(stopwords=STOPWORDS).generate(jd)
not_jd_cloud = WordCloud(stopwords=STOPWORDS).generate(not_jd)


plt.figure(1)
plt.imshow(jd_cloud)
plt.title('jd_cloud')
plt.axis("off")
plt.figure(2)
plt.imshow(not_jd_cloud)
plt.title('not_jd_cloud')
plt.axis("off")
```




    (-0.5, 399.5, 199.5, -0.5)




![Imgur](https://i.imgur.com/L1SlVyc.png)



![Imgur](https://i.imgur.com/QTWJIlM.png)


# CountVectorization with SVM


```python
svm_C = make_pipeline(CountVectorizer(ngram_range=(1,2)),SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=5, random_state=42), ).fit(train_posts, train_tags)

svm_prediction = svm_C.predict(test_posts)
SVM_score_train = f1_score(train_tags, svm_C.predict(train_posts), average='macro')
SVM_score_test = f1_score(test_tags, svm_C.predict(test_posts), average='macro')

print 'SVM_score_f1(test):{}, SVM_score_f1(train):{}'.format(SVM_score_test, SVM_score_train) 
```

    SVM_score_f1(test):0.782114278382, SVM_score_f1(train):0.851741742302


# TFIDF with LR



```python
lr1 = make_pipeline(CountVectorizer(ngram_range=(1,2)), LogisticRegression(), ).fit(train_posts, train_tags)

lr1_prediction = lr1.predict(test_posts)
lr1_score_train = f1_score(train_tags, lr1.predict(train_posts), average='macro')
lr1_score_test = f1_score(test_tags, lr1.predict(test_posts), average='macro')


print 'lr1_score_f1(test):{} --- lr1_score_f1(train):{}'.format(lr1_score_test, lr1_score_train) 
```

    lr1_score_f1(test):0.82265061112 --- lr1_score_f1(train):0.936119391843



```python
cv = Pipeline([('cv', CountVectorizer(ngram_range=(1,2)))])
# All together
feats =  FeatureUnion([('cv', cv),('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])
lr2 = make_pipeline(feats, LogisticRegression(), ).fit(train_posts, train_tags)
lr2_prediction= lr2.predict(test_posts)

lr2_score_train = f1_score(train_tags, lr2.predict(train_posts), average='macro')
lr2_score_test = f1_score(test_tags, lr2.predict(test_posts), average='macro')

print 'lr_score_f1(test):{} --- lr_score_f1(train):{}'.format(lr2_score_test, lr2_score_train) 
```

    lr_score_f1(test):0.835346783124 --- lr_score_f1(train):0.946749878492


## Stemming 
Preprocessing the posts with stop word removal and then porter stemming improved the F1 macro scores of both logistic regression and XGBoost classifiers.


```python
stemmer = stem.PorterStemmer()
def porter_stem(sentence):
    stemmed_sequence = [stemmer.stem(word) for word in tokenizer.tokenize(sentence)]
    return ' '.join(stemmed_sequence)
```


```python
stemmed_train = [porter_stem(post) for post in train_posts]
stemmed_test = [porter_stem(post) for post in test_posts]
```

## Logistic regression with Porter temming
This is the best performing classifier according to the F1 macro score, (but not according to the micro or weighted scores). In this case, the addition of LDA topics(but not the ad-hoc text features) leads to an improvement in the classification performance. 



```python
lda = Pipeline([('tf', CountVectorizer()), ])
lda_tfidf_features =  FeatureUnion([('lda', lda),
                                    ('tfidf', TfidfVectorizer(strip_accents='unicode', min_df=4))])
stem_lr_model = make_pipeline(lda_tfidf_features, LogisticRegression(C=1.5, penalty='l1', random_state=0)
                             ).fit(stemmed_train, train_tags)
stem_lr_prediction = stem_lr_model.predict(stemmed_test)

print('training score:', f1_score(train_tags, stem_lr_model.predict(stemmed_train), average='macro'))
stem_lr_macro_f1 = f1_score(test_tags, stem_lr_prediction, average='macro')
print('testing score:', stem_lr_macro_f1)
```

    ('training score:', 0.89063034571239386)
    ('testing score:', 0.81463428191661325)


## XGboost with Stemming
In the case of the XGBoost classifier the LDA topic and other ad-hoc text features did not seem to improve the performance. The reg_alpha parameter seems usually set to 0 for XGBoost tunings. Here setting it to 4 worked as a way to decrease the risk of overfitting.


```python
stem_xgb_model = make_pipeline(TfidfVectorizer(strip_accents='unicode', min_df=5, ngram_range=(1, 2)), 
                               xgb.XGBClassifier(max_depth=10, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, 
                                                 gamma=.01, reg_alpha=4, objective='multi:softmax', num_class= 2)
                              ).fit(stemmed_train, train_tags) 

stem_xgb_prediction = stem_xgb_model.predict(stemmed_test)
print('training score:', f1_score(train_tags, stem_xgb_model.predict(stemmed_train), average='macro'))
stem_xgb_macro_f1 = f1_score(test_tags, stem_xgb_prediction, average='macro')
print('testing score:', stem_xgb_macro_f1)
```

    ('training score:', 0.84150781846275979)
    ('testing score:', 0.79783721878070712)
    done in 16.184s.
                 precision    recall  f1-score   support
    
            0.0       0.84      0.94      0.89      5678
            1.0       0.83      0.62      0.71      2668
    
    avg / total       0.84      0.84      0.83      8346
    


### Majority Voting Ensembling
The results of the above classifiers are ensembled via a simple majority voting scheme. In case of no majority, the prediction of the logistic regression without stemming (i.e. the classifier with the highest macro F1 score) is picked.


```python
def majority_element(a):
    c = Counter(a)
    value, count = c.most_common()[0]
    if count > 1:
        return value
    else:
        return a[1]

merged_predictions = [[s[0],s[1],s[2]] for s in zip(stem_lr_prediction, lr2_prediction, stem_xgb_prediction)]
majority_prediction = [majority_element(p) for p in merged_predictions]

print('majority vote ensemble:', f1_score(test_tags, majority_prediction, average='macro')) 
print(classification_report(test_tags, majority_prediction))
```

    ('majority vote ensemble:', 0.82621301191918228)
                 precision    recall  f1-score   support
    
            0.0       0.87      0.93      0.90      5678
            1.0       0.83      0.69      0.75      2668
    
    avg / total       0.85      0.86      0.85      8346
    


### Comparison of different F1 score metrics for the classifiers
The score comparison below shows that micro and weighted F1 scores are generally similar for the same classifier, unlike macro F1 score. In the case of micro F1 score, the baseline classifier is relatively close to the other classifiers performance wise. Besides, the best classifier for the micro/weighted F1 scores is not the top classifier on macro F1 score.


```python
classifiers = ['svm_prediction','lr1_prediction', 'lr2_prediction','stem_lr_prediction', 'stem_xgb_prediction', 'majority_prediction']
predictions = (svm_prediction, lr1_prediction, lr2_prediction, stem_lr_prediction, stem_xgb_prediction, majority_prediction)
for pred, clfs in zip(predictions, classifiers):
    print(''.join((clfs,':')))
    print('macro:',f1_score(test_tags, pred, average='macro'))
    print('weighted:',f1_score(test_tags, pred, average='weighted'))
    print('micro:',f1_score(test_tags, pred, average='micro'))
    print()
```

    svm_prediction:
    ('macro:', 0.78211427838194847)
    ('weighted:', 0.81922034679102473)
    ('micro:', 0.83069734004313445)
    ()
    lr1_prediction:
    ('macro:', 0.82265061111980187)
    ('weighted:', 0.84992577816889392)
    ('micro:', 0.8549005511622334)
    ()
    lr2_prediction:
    ('macro:', 0.83534678312358779)
    ('weighted:', 0.85922500080596842)
    ('micro:', 0.86196980589503958)
    ()
    stem_lr_prediction:
    ('macro:', 0.81463428191661325)
    ('weighted:', 0.84214905238089888)
    ('micro:', 0.8460340282770189)
    ()
    stem_xgb_prediction:
    ('macro:', 0.79783721878070712)
    ('weighted:', 0.83014282130292261)
    ('micro:', 0.83752695902228613)
    ()
    majority_prediction:
    ('macro:', 0.82621301191918228)
    ('weighted:', 0.85220434714428583)
    ('micro:', 0.85609872993050562)
    ()

