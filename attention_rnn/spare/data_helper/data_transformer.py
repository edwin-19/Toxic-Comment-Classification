import numpy as np 
import re

from config import dataset_config

from keras.preprocessing.text import Tokenizer
from data_helper import data_loader

class DataTransformer(object):
    def __init__(self, max_num_words, max_sequence_length, char_level):
        self.data_loader = data_loader.DataLoader()
        # self.clean_word_dict = self.data_loader.load_clean_words()
        self.train_df = self.data_loader.load_dataset(dataset_config.TRAIN_PATH)
        self.test_df = self.data_loader.load_dataset(dataset_config.TEST_PATH)
        
        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.char_level = char_level
        self.tokenizer = None
    
    def prepare_data(self):
        # Number of classes to predict
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        # Since we cleaned data beforehand we dont need to clean again
        sentences_train = self.train_df['comment_text'].values
        sentences_test = self.test_df['comment_text'].values
        
        print("Doing preprocessing...")
        
        self.comments = sentences_train
        self.test_comments = sentences_test
        
        self.build_tokenizer(self.comments + self.test_comments)
        train_sequences = self.tokenizer.texts_to_sequences(self.comments)
        training_labels = self.train_df[list_classes].values
        test_sequences = self.tokenizer.texts_to_sequences(self.test_comments)
        
        print("Preprocessed.")
        
        return train_sequences, training_labels, test_sequences
    
    def build_embedding_matrix(self, embedding_index):
        nb_words = min(self.max_num_words, len(embedding_index))
        embedding_matrix = np.zeros((nb_words, 300))
        word_index = self.tokenizer.word_index
        null_words = open('dataset/null-word.txt', 'w', encoding='utf-8')
        
        for word, i in word_index.items():
            if i >= self.max_num_words:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
                continue
            
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
        
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        return embedding_matrix
    
    def build_tokenizer(self, comments):
        self.tokenizer = Tokenizer(num_words=self.max_num_words, char_level=self.char_level)
        self.tokenizer.fit_on_texts(comments)
        

