# Import pyhton library
import numpy as np
import pandas as pd
import nltk
from keras.preprocessing import sequence, text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import warnings
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import re
from nltk import FreqDist
from keras.callbacks import ModelCheckpoint


# import data set
train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
sample_sub = pd.read_csv('../input/sampleSubmission.csv')

# print statistics about the dataset
print('There are %d training data.', train.shape[0])
print('There are %s testing data.', test.shape[0])




## clean the phrases by converting to lower case,
lemma = WordNetLemmatizer()
def clean_review(review_col):
    review_corpus = []
    for i in range(0, len(review_col)):
        review = str(review_col[i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = [lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review = ' '.join(review)
        review_corpus.append(review)
    return np.array(review_corpus)


#change the labels to vector and create training, validation and test data
y = to_categorical(train.Sentiment.values, num_classes=5)
X_train, X_val, y_train, y_val = train_test_split(clean_review(train.Phrase.values), y,
                                                  test_size=0.20, random_state=42)
X_test = clean_review(test.Phrase.values)
y_test = to_categorical(sample_sub.Sentiment.values, num_classes=5)
# print statistics about the dataset
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)



all_words = ' '.join(X_train)
all_words = word_tokenize(all_words)
dist = FreqDist(all_words)
num_unique_word = len(dist)

r_len = []
for text in X_train:
    word = word_tokenize(text)
    l = len(word)
    r_len.append(l)

MAX_REVIEW_LEN = np.max(r_len)

max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 256
epochs = 10
num_classes = y.shape[1]

#converting text to sequences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

#pad the sequences to max length
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape, X_val.shape, X_test.shape)

#creating the LSTM based model
model = Sequential()
model.add(Embedding(max_features, 128, mask_zero=True))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.3, return_sequences=False))
model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# adding the check pointer
checkpointer = ModelCheckpoint(filepath='weights.best.hdf5',
                               verbose=1, save_best_only=True)
print("fitting the model")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10,
                    batch_size=batch_size, verbose=1, callbacks=[checkpointer])
print("fitting done")

#load the best weight
model.load_weights('weights.best.hdf5')

#calculate the final accuracy on test set
score, accuracy = model.evaluate(X_test, y_test)
print(accuracy)