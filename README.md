<h1 align="center">Quora Question Pairs</h1>

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.


## Problem Statement

Identify which questions asked on Quora are duplicates of questions that have already been asked. This could be useful to instantly provide answers to questions that have already been answered. 

The task is to predict whether a pair of questions are duplicates or not. It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.


## Kaggle dataset

**Link**: https://www.kaggle.com/c/quora-question-pairs

The Kaggle dataset consists of the following columns:

- id - the id of a training set question pair
- qid1, qid2 - unique ids of each question (only available in train.csv)
- question1, question2 - the full text of each question
- is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.


## Performance Metrics

Predictions are evaluated on the following metrics between the predicted values and the ground truth.

- Log Loss (https://www.kaggle.com/wiki/LogarithmicLoss)
- Binary Confusion Matrix


### Feature extraction

- Number of characters in question1 and question2
- Number of words in question1 and question2
- Number of characters in question1 and question2 (removing whitespaces)
- Difference between number of characters in question1 and question2
- Difference between number of words in question1 and question2
- Difference between number of characters in question1 and question2 (removing whitespaces)
- Number of common words in question1 and question2
- Common words ratio i.e. Number of common words in question1 and question2 / Total number of words in question1 and question2


### Text Preprocessing

- Remove HTML tags
- Remove extra whitespaces
- Convert accented characters to ASCII characters
- Expand contractions
- Remove special characters
- Lowercase all texts
- Remove stopwords
- Lemmatization

## Advanced Feature Extraction

#### FuzzyWuzzy

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


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/maanavshah/quora-question-pairs. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.


## License

The content of this repository is licensed under [MIT LICENSE](LICENSE).
