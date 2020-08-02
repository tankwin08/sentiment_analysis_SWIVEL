# Sentinel analysis - SWIVEL

![alt text](https://github.com/tankwin08/sentiment_analysis_SWIVEL/blob/master/figs/nlp_flowchart.png)

## Objetive

To retrain the pretrained model (Submatrix-wise Vector Embedding Learner (SWIVEL) using using a small collected review datasets and classify the reviews of customer feedback as either positive or negative. 

## Data

The data for this project can be downloaded from [here](https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/).

Specificall, these data were used for our training data with two categories: postive and negative.


## Basic introduction of NLP and SWIVEL
Genealy, it require an amount of data to train and a long time to train a NLP model for a specific dataset. Transfer learning is commonly used in this case to conduct the sentiment analsis for Natural Language Processing (NLP) problem. 

Swivel performs approximate factorization of the point-wise mutual information matrix via stochastic gradient descent.
It uses a piecewise loss with special handling for unobserved co-occurrences, and thus makes use of all the information in the matrix.


The codes and framework are implemented in Keras. For a quick start, you can find the following tutorials:

* Shazeer, Noam, Ryan Doherty, Colin Evans, and Chris Waterson. "Swivel: Improving embeddings by noticing what's missing." arXiv preprint arXiv:1602.02215 (2016).

* [Deep Transfer Learning for Natural Language Processing: Text Classification with Universal Embeddings](https://towardsdatascience.com/deep-transfer-learning-for-natural-language-processing-text-classification-with-universal-1a2c69e5baa9)
* [Keras Tutorial: How to Use Google's Universal Sentence Encoder for Spam Classification](http://hunterheidenreich.com/blog/google-universal-sentence-encoder-in-keras/)

These examples make use of TensorFlow Hub, which allows pretrained models to easily be loaded into TensorFlow.

###  [Pretrained Word Embeddings](https://www.analyticsvidhya.com/blog/2020/03/pretrained-word-embeddings-nlp/)

Pretrained Word Embeddings are the embeddings learned in one task that are used for solving another similar task.

These embeddings are trained on large datasets, saved, and then used for solving other tasks. That’s why pretrained word embeddings are a form of Transfer Learning.

CBOW (Continuous Bag Of Words) and Skip-Gram are two most popular frames for word embedding. In CBOW the words occurring in context (surrounding words) of a selected word are used as inputs and middle or selected word as the target. Its the other way round in Skip-Gram, here the middle word tries to predict the words coming before and after it.

**Why do we need Pretrained Word Embeddings?**

Pretrained word embeddings capture the semantic and syntactic meaning of a word as they are trained on large datasets. They are capable of boosting the performance of a Natural Language Processing (NLP) model. These word embeddings come in handy during hackathons and of course, in real-world problems as well.

But why should we not learn our own embeddings? Well, learning word embeddings from scratch is a challenging problem due to two primary reasons:

* 1 Sparsity of training data

* 2 Large number of trainable parameters


### [Co-occurrence matrix](https://towardsdatascience.com/word2vec-made-easy-139a31a4b8ae)
A co-occurrence matrix tells us how often a particular pair of words occur together. Each value in a co-occurrence matrix is a count of a pair of words occurring together.
 
Generally, a co-occurrence matrix will have specific entities in rows (ER) and columns (EC). The purpose of this matrix is to present the number of times each ER appears in the same context as each EC. As a consequence, in order to use a co-occurrence matrix, you have to define your entites and the context in which they co-occur.

You can refer to more details [here](https://iksinc.online/tag/co-occurrence-matrix/)






