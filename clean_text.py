import pandas as pd
import nltk
from nltk import TweetTokenizer, WordNetLemmatizer
import re
import time
import os

from filter import bag_of_words

lemmatizer = WordNetLemmatizer() 
tokenizer = TweetTokenizer()

def clean_text(text):
    text = text.lower()
    
    for selection in bag_of_words:
        text = re.sub(selection[0], selection[1], text)
    
    toknized_text = tokenizer.tokenize(text)
    lemetized = []
    
    for word, tag in nltk.pos_tag(toknized_text):
        if tag == "NN":
            # NN: noun, common, singular or mass
            lemetized.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag == "VB":
            # VB: verb, base form
            lemetized.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag == "R":
            # JJ: adjective or numeral, ordinal
            lemetized.append(lemmatizer.lemmatize(word, pos='r'))
        elif tag == "JJ":
            # R: adverb
            lemetized.append(lemmatizer.lemmatize(word, pos='a'))
        else:
            lemetized.append(word)
            
    return " ".join(lemetized)

if __name__ == "__main__":
    cleaned_path = 'data/toxic_comment/cleaned'
    if not os.path.exists(cleaned_path):
        os.makedirs(cleaned_path)
        
    train_df = pd.read_csv('data/toxic_comment/train.csv')
    test_df = pd.read_csv('data/toxic_comment/test.csv')
    
    # fill empty columns
    print("Cleaning data")
    train_df['comment_text'].fillna('empty comment')
    test_df['comment_text'].fillna('empty comment')
    
    start_time = time.time()
    cleaned_corpus = []
    for c in train_df.comment_text:
        cleaned_corpus.append(clean_text(c))
        
    train_df['comment_text'] = cleaned_corpus
    train_df.to_csv(cleaned_path + '/train.csv')
    print("Total time to clean train data: ", time.time() - start_time)
    
    start_time = time.time()
    cleaned_corpus = []
    for c in test_df.comment_text:
        cleaned_corpus.append(clean_text(c))

    test_df['comment_text'] = cleaned_corpus
    test_df.to_csv(cleaned_path + '/test.csv')
    print("Total time to clean test data: ", time.time() - start_time)
    
    