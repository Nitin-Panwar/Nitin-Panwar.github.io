---
layout: post
title: Top 10 Elastic Search Interview Questions
published: True
---

We have ofen see people struggling with Elastic search basic interview question. In this post i will try to give you what are the top
10 Elastic search interview question, that you should know before going for any interview. Before that i would highly recommond you to 
check [this](http://nitin-panwar.github.io/Elasticsearch-tutorial-for-beginners-using-Python/) blog, in this blog post i gave basic 
uderstanding of Elastic search. 

So let'g get started...


### 1.  What are document,index,indexing and inverted index in Elastic serarch?
#### Document: 
A document in Elastic search can be considered as a row in relational database. A document might have different data types. 

#### Index: 
An Index in Elastic search can be considered as a table in relational database. An Index will have multiple documentes. It is 
not nessery to have all the documents with same schema, we can force same schema but it is not mendaory like in relational 
databases, where we have difiened schema at table level. 

#### Indexing: 
Inserting documants in index is called as indexing. 

#### Inverted Index:
Inverted Index is backbone of Elasticsearch which make full-text search fast.Inverted index consists of a list of all unique words that occurs in documents and for each word, maintain a list of documents number and positions in which it appears.

For Example: There are two documents and having content as:
1: FacingIssuesOnIT is for ELK.
2: If ELK check FacingIssuesOnIT.

To make inverted index each document will split in words and create below sorted index.
[](https://i.imgur.com/loapMVe.png)

Now when we do some full-text search for String will sort documents based on existence and occurrence of matching counts.
Usually in Books we have inverted indexes on last pages. Based on the word we can thus find the page on which the word exists.




### 2. What are shards and replicas in Elastic Search?
