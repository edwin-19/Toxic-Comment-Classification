import pandas as pd 
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from attention_rnn.keras.embeddings import load_embedding

class DataLoader(object):
    def __init__(self, train_df_path, test_df_path, max_num_words, max_sequence_length, embedding_dim):
        self.train_df = pd.read_csv(train_df_path)
        self.test_df = pd.read_csv(test_df_path)
        
        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        
        self.list_classes = ["toxic", "severe_toxic", 
                             "obscene", "threat", 
                             "insult", "identity_hate"]
        
        self.train_df['bad_comment'] = self.train_df["toxic"]\
             | self.train_df["severe_toxic"] | self.train_df["obscene"]\
                  | self.train_df["threat"] | self.train_df["insult"]\
                       | self.train_df["identity_hate"]
        
    def prepare_data(self):
        self.y = self.train_df[self.list_classes].values
        self.comments = [text for text in self.train_df['comment_text'].fillna("no comment").values]
        self.test_comments = [text for text in self.test_df['comment_text'].fillna("no comment").values]
        self.build_tokenizer(self.comments + self.test_comments)
        
        self.sequences = self.tokenizer.texts_to_sequences(self.comments)
        self.test_sequences = self.tokenizer.texts_to_sequences(self.test_comments)
        self.word_index = self.tokenizer.word_index
        
        print('Found %s unique tokens' % len(self.word_index))
        
        self.data = pad_sequences(self.sequences, maxlen=self.max_sequence_length)
        print('Shape of data tensor:', self.data.shape)
        print('Shape of label tensor:', self.y.shape)
        
        self.test_data = pad_sequences(self.test_sequences, maxlen=self.max_sequence_length)
        print('Shape of test_data tensor:', self.test_data.shape)
        
        # Prepare embeddings matrix
        print('Preparing embedding matrix')
        self.nb_words = min(self.max_num_words, len(self.word_index))

        self.embedding_matrix = np.zeros((self.nb_words, self.embedding_dim))
    
    def load_embeddings(self, embedding_path):
        return load_embedding(embedding_path)

    def build_tokenizer(self, comments):
        self.tokenizer = Tokenizer(num_words=self.max_num_words)
        self.tokenizer.fit_on_texts(comments)