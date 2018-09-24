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

### 1. What Is Elasticsearch ?

Elasticsearch is a search engine based on Lucene. It provides a distributed, multitenant-capable full-text search engine with an HTTP web interface and schema-free JSON documents. Elasticsearch is developed in Java and is released as open source under the terms of the Apache License.


### 2.  What are document,index,indexing and inverted index in Elastic serarch?
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
![table](https://i.imgur.com/loapMVe.png)

Now when we do some full-text search for String will sort documents based on existence and occurrence of matching counts.
Usually in Books we have inverted indexes on last pages. Based on the word we can thus find the page on which the word exists.




### 3. What are shards and replicas in Elastic Search?

#### Shard:-
Due to resource limitations like RAM, vCPU etc, for scale-out, applications need to employ multiple instances of ElasticSearch on separate machines. Data in an index can be divided into multiple partitions, each handled by a separate node (instance) of ElasticSearch. Each such partition is called a shard. By default an ElasticSearch index has 5 shards.

#### Replica:-
Each shard in ElasticSearch has 2 copy of the shard. These copies are called replicas. They serve the purpose of high-availability and fault-tolerance.


### 4. What Is The Query Language Of Elasticsearch ?

ElasticSearch uses the Apache Lucene query language, which is called Query DSL.

### 5. 4. What is a Node in Elasticsearch?

A node is an important component in an Elasticsearch which is needed before starting an instance of Elasticsearch. A group of nodes is called a cluster. If a single node of Elasticsearch is running, then it is called a cluster of one node. In networking, the transport layer is used to establish communication between nodes of a cluster. Each and every node existing in a cluster can send client requests to each other and can establish communication with each other.

There are several types of nodes such as 

###### Master node:-
A Master node is a node that controls the entire cluster. 

###### Data node:-

A data node is a node that holds data in it and performs logical operations on the data.

###### Ingest node:-

An Ingest node is a node that can be used to ingest pipeline which means a series of processors to a document to perform some transformations before indexing the document. 

###### Tribe node:-
A Tribe node is a node that performs some coordination to connect to multiple clusters across all the connected clusters and perform some logical operations or searches. 

By default, a node will always be a master node and a data node, but depending on the large requirements, node configurations should be carried out.



### 5. What Is A Cluster In Elasticsearch ?

Cluster is a collection of one or more nodes (servers) that together holds your entire data and provides federated indexing and search capabilities across all nodes. A cluster is identified by a unique name which by default is "elasticsearch". This name is important because a node can only be part of a cluster if the node is set up to join the cluster by its name.


















