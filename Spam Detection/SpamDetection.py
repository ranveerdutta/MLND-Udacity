
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. load dataset
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('C:/Users/ranveer/Downloads/smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Output printing out first 5 columns
#print(df.head())

# 2. pre-processing, change category label to number from String, otherwise it may create problem with scikit learn
df['label'] = df.label.map({'ham':0, 'spam':1})
#print(df.shape)
#print(df.head())

# split into training and testing sets
# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)


print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)




# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

#fit the training data against Naive Bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

#validate against test data
predictions = naive_bayes.predict(testing_data)


#calculate accuracy, precision, recall and F1 score
# Precsion = [True Positives/(True Positives + False Positives)]
# Recall = [True Positives/(True Positives + False Negatives)]
# F1 score is weighted average of the precision and recall scores.
# This score can range from 0 to 1, with 1 being the best possible F1 score.
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))