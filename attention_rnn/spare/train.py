import os
import re
import csv
import codecs
import numpy as np 
import pandas as pd 
import operator
import sys

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from models.keras import model
from models.keras import trainer
from data_helper.data_loader import DataLoader
from matplotlib import pyplot as plt

path = 'data/toxic_comment/cleaned/'
embedding_file = 'features/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

train_data_file = path + 'train.csv'
test_data_file = path + 'test.csv'

max_sequence_length = 400
max_nb_words = 100000
embedding_dim = 300

train_df = pd.read_csv(train_data_file)
test_df = pd.read_csv(test_data_file)
data_loader = DataLoader()
embedding_index = data_loader.load_embedding(
    embedding_file
)

train_df['bad_comment'] = train_df["toxic"] | train_df["severe_toxic"] | train_df["obscene"] | train_df["threat"] | train_df["insult"] | train_df["identity_hate"]

print('Processing text dataset')
list_sentences_train = train_df["comment_text"].fillna("no comment").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "bad_comment"]
y = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("no comment").values

comments = []
for text in list_sentences_train:
    comments.append(text)

test_comments=[]
for text in list_sentences_test:
    test_comments.append(text)

tokenizer = Tokenizer(num_words=max_nb_words)

tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=max_sequence_length)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)
print('Shape of test_data tensor:', test_data.shape)

# Prepare embeddings matrix
print('Preparing embedding matrix')
nb_words = min(max_nb_words, len(word_index))

embedding_matrix = np.zeros((nb_words, embedding_dim))

null_count = 0
for word, i in word_index.items():
    if i >= max_nb_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        null_count += 1
print('Null word embeddings: %d' % null_count)

def get_model():
    return model.get_av_rnn(
        nb_words, embedding_dim, embedding_matrix, max_sequence_length,
        out_size=7
    )

keras_model_trainer = trainer.KerasModelTrainer(
    model_stamp='hier_avrnn', epoch_num=50
)

models, val_loss, total_auc, fold_predictions = keras_model_trainer.train_folds(
    data, y, fold_count=10, batch_size=256, get_model_func=get_model
)

print("Overall val-loss:", val_loss, "AUC", total_auc)