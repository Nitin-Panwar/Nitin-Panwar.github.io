---
layout: post
title: Training Spacy matcher for Location extraction 
published: true
---
Here we go... 
If you want to extract location from a sentence, then below solution will help you to do so. 

As you know NER(Named Entity Recognition) works well if you are dealing with some Internationl location, But if your task is to extract local location from a sentence then NER wouldn't work or you have to train NER for the local locations as well. But if you are having a limited number of locations and you want to extract it from the sentence then give a try to Spacy Matcher.

First you have to train it with all the availble location then it will do the extraction magic for you.


```python
# Load required modules
from spacy.matcher import Matcher
from spacy.attrs import IS_PUNCT, LOWER
import spacy

nlp = spacy.load('en')
matcher = Matcher(nlp.vocab)
```

#### There is a specific pattern to train Sapcy Matcher- 

#### E.g pattern = {'HelloWorld': [{'LOWER': 'hello'}, {'LOWER': 'world'}]}


```python
def skillPattern(skill):
    pattern = []
    for b in skill.split():
        pattern.append({'LOWER':b})  
    return pattern

def buildPatterns(skills):
    pattern = []
    for skill in skills:
        pattern.append(skillPattern(skill))
    return list(zip(skills, pattern))
```


```python
def on_match(matcher, doc, id, matches):
    return matches

def buildMatcher(patterns):
    for pattern in patterns:
        matcher.add(pattern[0], on_match, pattern[1])
    return matcher
```


```python
def cityMatcher(matcher, text):
    skills = []
    doc = nlp(unicode(text.lower()))
    matches = matcher(doc)
    for b in matches:
        match_id, start, end = b
        print doc[start : end]
```


```python
cities = [ u'delhi',
u'bengaluru',
u'kanpur',
u'noida',
u'ghaziabad',
u'chennai',
u'hydrabad',
u'luckhnow',
u'saharanpur',
u'dehradun',
u'bombay']
```


```python
patterns = buildPatterns(cities)
```


```python
city_matcher = buildMatcher(patterns)
### Size of dictionary 
len(city_matcher)
```




    11




```python
cityMatcher(city_matcher, "I am from Saharanpur, i live in bengaluru..")
```

    saharanpur
    bengaluru
