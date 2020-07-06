<h1 align="center">Quora Question Pairs</h1>

Quora is a question-and-answer website where questions are asked, answered, edited, and organized by its community of users in the form of opinions.

In September 2018, Quora reported hitting 300 million monthly users. With over 300 million people visit Quora every month, it’s no surprise that many people ask duplicated questions, that is, questions that have the same intent. For example, questions like “How can I be a good geologist?” and “What should I do to be a great geologist?” are duplicate questions because they all have the same intent and should be answered once and once only.


## Problem Statement

Identify which questions asked on Quora are duplicates of questions that have already been asked. This could be useful to instantly provide answers to questions that have already been answered. 

The task is to predict whether a pair of questions are duplicates or not. It is a binary classification problem, for a given pair of questions we need to pr

We will develop a system to classify whether question pairs are duplicates or not. We start by information retrieval with the help of models like BOW, TF-IDF, or Word2Vec. We use SGD and XGBoost for classification.


## Kaggle dataset

**Link**: https://www.kaggle.com/c/quora-question-pairs

The Kaggle dataset consists of the following columns:

- **id** - the id of a training set question pair
- **qid1, qid2** - unique ids of each question (only available in train.csv)
- **question1, question2** - the full text of each question
- **is_duplicate** - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.


## Performance Metrics

Predictions are evaluated on the following metrics between the predicted values and the ground truth.

- Log Loss (https://www.kaggle.com/wiki/LogarithmicLoss)
- Binary Confusion Matrix


## Exploratory Analysis

The distribution of duplicate and non-duplicate question pairs is:

<div align="center">
  <img src="images/is_duplicate_dist.png">
</div>


The distribution of questions among the data points is as follows. We can see that around **63 %** questions pair are not duplicate and **36 %** questions pair are duplicate.

<div align="center">
  <img src="images/quest_dist.png">
</div>


Total number of questions: **808702**
Total number of unique questions: **537388**
Total number of duplicate questions occuring more than once: **20.82 %**
Maximum occurence of single repeated question: **161**

The following is the log-histogram of the questions frequency:

<div align="center">
  <img src="images/log_hist.png">
</div>


## Feature extraction

We have constructed few features like:

- Number of characters in question1 and question2
- Number of words in question1 and question2
- Number of characters in question1 and question2 (removing whitespaces)
- Difference between number of characters in question1 and question2
- Difference between number of words in question1 and question2
- Difference between number of characters in question1 and question2 (removing whitespaces)
- Number of common words in question1 and question2
- Common words ratio i.e. Number of common words in question1 and question2 / Total number of words in question1 and question2

Let's take a look at plot of common words ratio, we can observe that distibution for non-duplicate and duplicate question pairs can be useful as they are neither overlapping completely nor separated apart ideally.

<div align="center">
  <img src="images/common_word_dist.png">
</div>


## Text Preprocessing

You may have noticed that we have off a lot work to do in terms of text cleaning. After some inspections, a few tries and ideas from https://www.kaggle.com/currie32/the-importance-of-cleaning-text, I decided to clean the text as follows:

- Remove HTML tags
- Remove extra whitespaces
- Convert accented characters to ASCII characters
- Expand contractions
- Remove special characters
- Lowercase all texts
- Remove stopwords
- Lemmatization

## Advanced Feature Extraction

### FuzzyWuzzy

The following text ratio can be extracted from the fuzzywuzzy library.

- Fuzz Ratio
- Fuzz Parial Ratio
- Token Sort Ratio
- Token Set Ratio

**Credits:**
<br>
https://github.com/seatgeek/fuzzywuzzy
<br>
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

The following is the distribution plot for all FuzzyWuzzy features:


<div align="center">
  <img src="images/fuzz_ratio.png">
  <img src="images/fuzz_partial_ratio.png">
</div>

<div align="center">
  <img src="images/token_sort_ratio.png">
  <img src="images/token_set_ratio.png">
</div>


## Feature Analysis

Word Clouds help us to understand how frequency of words can contribute to identifying duplicate question pairs. The bigger the word means the more number of occurences of the word.

Word cloud for non duplicate question pairs:

<div align="center">
  <img src="images/word_cloud_nondup.png">
</div>

Word cloud for duplicate question pairs:

<div align="center">
  <img src="images/word_cloud_duplicate.png">
</div>



### Machine Learning Models:
   - Trained a random model to check Worst case log loss and got log loss as 0.887699
   - Trained some models and also tuned hyperparameters using Random and Grid search. I didnt used total train data to train my algorithms. Because of ram availability constraint in my PC, i sampled some data and Trained my models. below are models and their logloss scores. you can check total modelling and feature extraction [here](https://github.com/UdiBhaskar/Quora-Question-pair-similarity/blob/master/Quora%20Question%20pair%20similarity.ipynb)  
   For below table BF - Basic features, AF - Advanced features, DF - Distance Features including WMD.

| Model         | Features Used | Log Loss |
| ------------- | ------------- | ------------- |
| Logistic Regression  | BF + AF  | 0.4003415  |
| Linear SVM           | BF + AF  | 0.4036245  |
| Random Forest  | BF + AF  | 0.4143914  |
| XGBoost  | BF + AF  | 0.362546  |
| Logistic Regression  | BF + AF + Tf-Idf  | 0.358445  |
| Linear SVM  | BF + AF + Tf-Idf  | 0.362049  |
| Logistic Regression  | BF + AF + DF + AVG-W2V  | 0.3882535  |
| Linear SVM  |  BF + AF + DF + AVG-W2V  | 0.394458  |
| XGBoost  | BF + AF + DF + AVG-W2V  | 0.313341  |



## Solution

BOW, TF-IDF and Word2venc are the most common methods people use in information retrieval. Generally speaking, SVMs and Naive Bayes are more common for classification problem, however, We will be using the SGD and XGBoost classifiers that allow us to specify the log loss metric while training the model.

**XGBoost** is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. **SGD** implements regularized linear models with stochastic gradient descent (SGD) learning i.e the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). 


### Bag-Of-Words (CountVectorizer)

This following strategy is called the Bag of Words. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document. It tokenizes the documents and counts the occurrences of token and return them as a sparse matrix.


### TF-IDF (Word Level and N-gram Level)

The tf–idf is the product of two statistics, term frequency and inverse document frequency, it is one of the most popular term-weighting schemes today. 

### Word2vec

Word2vec has several advantages over bag of words and TF-IDF scheme. Word2vec retains the semantic meaning of different words in a document. The context information is not lost. Another great advantage of Word2vec approach is that the size of the embedding vector is very small.

Now we will apply SGD and XGBoost classifiers to all three information retrieval models and check which of the gives us better performance.


We can see the log losses obtailed for different learning rates for SGD Clasiifier. We have selected the final learning rate as the one that has the lowest log loss value.

<div align="center">
    <figure>
      <img src="images/sgd_log_loss.png">
      <figcaption>Log Loss for BOW</figcaption>
    </figure>
    <figure>
      <img src="images/tfidf_log_loss.png">
      <figcaption>Log Loss for TF-IDF (Word Level)</figcaption>
    </figure>
    <figure>
      <img src="images/ngram_log_loss.png">
      <figcaption>Log Loss for TF-IDF (N-gram Level)</figcaption>
    </figure>
    <figure>
      <img src="images/word2vec_log_loss.png">
      <figcaption>Log Loss for Word2vec</figcaption>
    </figure>
</div>

Following are the confusion matrix for each model and classifier:

<div align="center">
    <figure>
      <img src="images/bow_sgd_cm.png">
      <figcaption>Confusion Matrix for BOW (SGD)</figcaption>
    </figure>
    <figure>
      <img src="images/bow_xgb_cm.png">
      <figcaption>Confusion Matrix for BOW (XGBoost)</figcaption>
    </figure>
    <figure>
      <img src="images/tf_sgd_cm.png">
      <figcaption>Confusion Matrix for TF-IDF Word Level (SGD)</figcaption>
    </figure>
    <figure>
      <img src="images/tf_xgb_cm.png">
      <figcaption>Confusion Matrix for TF-IDF Word Level (XGBoost)</figcaption>
    </figure>
    <figure>
      <img src="images/ngram_sgd_cm.png">
      <figcaption>Confusion Matrix for TF-IDF N-gram Level (SGD)</figcaption>
    </figure>
    <figure>
      <img src="images/ngram_xgb_cm.png">
      <figcaption>Confusion Matrix for TF-IDF N-gram Level (XGBoost)</figcaption>
    </figure>
    <figure>
      <img src="images/word2vec_sgd_cm.png">
      <figcaption>Confusion Matrix for Word2vec (SGD)</figcaption>
    </figure>
    <figure>
      <img src="images/word2vec_cm.png">
      <figcaption>Confusion Matrix for Word2vec (XGBoost)</figcaption>
    </figure>
</div>

We then selected the best model i.e. TF-IDF Word Level from all the available combinations and applied hypertuning using GridSearchCV. This helped us to get the best parameters for the estimator.

<div align="center">
    <figure>
      <img src="images/cm.png">
      <figcaption>Confusion Matrix for TF-IDF Word Level (XGBoost Hypertuning)</figcaption>
    </figure>
</div>

Our highest validation score is 80.7%, not bad at all for starters!
So far our best Xgboost model is word level TF-IDF + Xgboost, the recall of duplicate questions, that is, the proportion of duplicate questions that our model is able to detect over the total amount of duplicate questions is 0.79. And this is crucial for the problem at hand, we want to detect and eliminate as many duplicate questions as possible.


<div align="center">
  <img src="images/output.png">
</div>

With this in mind, we can develop hybrid of models (for example, Word2vec + TF-IDF and Xgboost model) to see whether this result can be improved.


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/maanavshah/quora-question-pairs. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.


## License

The content of this repository is licensed under [MIT LICENSE](LICENSE).
