from keras.models import load_model
from attention_rnn.keras.model import AttentionWeightedAverage
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import pandas as pd
import numpy as np 
import os

model = load_model('checkpoints/av_rnn/av_rnn16-0.03.h5', custom_objects={
    'AttentionWeightedAverage': AttentionWeightedAverage
})

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_SEQUENCE_LENGTH = 400
MAX_NB_WORDS = 100000

train_df = pd.read_csv('data/toxic_comment/cleaned/train.csv')
test_df = pd.read_csv('data/toxic_comment/cleaned/test.csv')
sample_submission = pd.read_csv('data/toxic_comment/sample_submission.csv')

comments = []
for text in train_df['comment_text'].fillna("no comment").values:
    comments.append(text)
    
test_comments=[]
for text in train_df['comment_text'].fillna("no comment").values:
    test_comments.append(text)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments + test_comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)
# test_sequences = tokenizer.texts_to_sequences([test_df['comment_text'][4]])

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

test_predicts = model.predict(test_data, batch_size=256, verbose=1)

# Submit = pd.DataFrame([test_df.id, test_df.comment_text],columns=['id', 'comment_text'])
if not os.path.exists('output'):
    os.makedirs('output')
    
Submit = test_df[['id', 'comment_text']].copy()
Submit2 = pd.DataFrame(test_predicts,columns=CLASSES)
Submit = pd.concat([Submit,Submit2],axis=1)
Submit.to_csv("output/avnn_rnn.csv",index=False)

# print(test_predicts)
# max_arg = np.argmax(np.squeeze(test_predicts, axis=0))
# print(CLASSES[max_arg])
# sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = test_predicts
# sample_submission.to_csv('submission18.csv', index=False)
