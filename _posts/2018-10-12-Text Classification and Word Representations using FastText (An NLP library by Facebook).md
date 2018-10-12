---
layout: post
title: Text Classification and Word Representations using FastText (An NLP library by Facebook)
published: False
---

FastText is a library created by the Facebook Research Team for efficient learning of word representations and sentence classification. Before moving forward with Fasttext, let's try to understand word representations. In very simplistic terms, Word Embeddings or Word representations are the texts converted into numbers and there may be different numerical representations of the same text. 

Take a look at this example – sentence=” Word Embeddings are Word converted into numbers ”

## Different types of Word Embeddings
The different types of word embeddings can be broadly classified into two categories-

### 1 Frequency based Embedding
There are generally three types of vectors that we encounter under this category.

#### 1.1 Count Vector

Let us understand this using a simple example.

D1: He is a lazy boy. She is also lazy.

D2: Neeraj is a lazy person.

The dictionary created may be a list of unique tokens(words) in the corpus =[‘He’,’She’,’lazy’,’boy’,’Neeraj’,’person’]

The count matrix M of size 2 X 6 will be represented as –

   He	She	lazy	boy	Neeraj	person
D1	1	 1	  2	   1	  0	      0
D2	0	 0	  1	   0	  1	      1

So the vectors for D1 and D2 will be [1,1,2,1,0,0] and [0,0,1,0,1,1] respectivelly.

#### 1.2 TF-IDF Vector

This is another method which is based on the frequency method but it is different to the count vectorization in the sense that it takes into account not just the occurrence of a word in a single document but in the entire corpus. Ideally, what we would want is to down weight the common words occurring in almost all documents and give more importance to words that appear in a subset of documents.

TF = (Number of times term t appears in a document)/(Number of terms in the document)

IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.


#### 1.3 Co-Occurrence Vector

Co-occurrence – For a given corpus, the co-occurrence of a pair of words say w1 and w2 is the number of times they have appeared together in a Context Window.

Context Window – Context window is specified by a number and the direction. 


### 2. Prediction based Embedding

Frequency based methods proved to be limited in their word representations until Mitolov etc. el introduced word2vec to the NLP community. These methods were prediction based in the sense that they provided probabilities to the words and proved to be state of the art for tasks like word analogies and word similarities. 

#### 2.1 CBOW (Continuous Bag of words)








