---
layout: post
title: Text Classification using machine learning
published: False
---

Near-duplicate detection: LSH is commonly used to deduplicate large quantities of documents, webpages, and other files. LSH refers to a family of functions (known as LSH families) to hash data points into buckets so that data points near each other are located in the same buckets with high probability, while data points far from each other are likely to be in different buckets. This makes it easier to identify observations with various degrees of similarity.

The problem of finding duplicate documents in a list may look like a simple task — use a hash table, and the job is done quickly and the algorithm is fast. However, if we need to find not only exact duplicates, but also documents with differences such as typos or different words, the problem becomes much more complex. In my case, I’ve had to find duplicates in a very long list of given questions.

After experimenting with string distance metrics (Levenshtein, Jaro-Winkler, Jaccard index) I’ve come to the conclusion that the Jaccard index performs sufficiently for this use case. The Jaccard index is an intersection over a union. We count the amount of common elements from two sets, and divide by the number of elements that belong to either the first set, the second set, or both. Let’s have a look at the example:

___“Who was the first ruler of Poland”___

___“Who was the first king of Poland”___

![Imgur](https://i.imgur.com/m8NGcHZ.png)

The size of the intersection is 6, while the size of the union is 6 + 1 + 1 = 8, thus the Jaccard index is equal to 6 / 8 = 0.75


We can conclude — the more common words, the bigger the Jaccard index, the more probable it is that two questions are a duplicate. So where we can set a threshold above which pairs would be marked as a duplicate? For now, let’s assume 0.5 as a threshold, but in a real life, we need to get this value by experiment. We could stop here, but current solution makes a number of comparisons growing quadratically (It’s 0.5*(n²-n) where n is the number of questions).

Choice of hashing function is tightly linked to the similarity metric we’re using. For Jaccard similarity the appropriate hashing function is min-hashing.

#### Min part of MinHash

It’s been shown earlier that the Jaccard can be a good string metric, however, we need to split each question into the words, then, compare the two sets, and repeat for every pair. The amount of pairs will grow rapidly. What if we somehow created a simple fixed-size numeric fingerprint for each sentence and then just compare the fingerprints?

To calculate MinHash we need to create the dictionary (a set of all words) from all our questions. Then, create a random permutation:

We have to iterate over the rows, writing the index in the respective cell, if the word being checked is present in the sentence.

![Imgur](https://i.imgur.com/UfTYWk6.png)

Now the second row:

![Imgur](https://i.imgur.com/e4pZaZf.png)

Only the first word occurrence is relevant (giving a minimal index — hence the name MinHash). 
We have a minimum value for all our questions and the first part of the fingerprint. To get the second one we need to create another random permutation and retrace our steps.








  


