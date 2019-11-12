import os
import re
import numpy as np 
from keras.callbacks import ModelCheckpoint, EarlyStopping

from attention_rnn.keras.model import get_av_rnn
from attention_rnn.keras.data_loader import DataLoader

def get_null_count(word_index, embedding_matrix, embedding_index, max_nb_words):
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
    
# Hyper-params
model_name = 'av_rnn'
ckpt_path = 'checkpoints/' + model_name + "/"
batch_size = 256
epochs = 50

# Training details
path = 'data/toxic_comment/cleaned/'
embedding_file = 'features/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

train_data_file = path + 'train.csv'
test_data_file = path + 'test.csv'

max_sequence_length = 400
max_nb_words = 100000
embedding_dim = 300

data_loader = DataLoader(
    train_data_file, test_data_file, 
    max_nb_words, max_sequence_length,
    embedding_dim
)

embedding_index = data_loader.load_embeddings(embedding_file)
data_loader.prepare_data()

get_null_count(
    data_loader.word_index, data_loader.embedding_matrix, 
    embedding_index, max_nb_words
)

# Set attention model here
model = get_av_rnn(
    data_loader.nb_words, embedding_dim,
    data_loader.embedding_matrix, max_sequence_length,
    out_size=len(data_loader.list_classes)
)

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

# Set callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
model_ckpt = ModelCheckpoint(
    ckpt_path + model_name + '{epoch:02d}-{loss:.2f}.h5', save_best_only=True, 
    monitor='loss', verbose=1
)
model.fit(
    data_loader.data, data_loader.y,
    batch_size=batch_size, epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping, model_ckpt]
)

with open(ckpt_path + 'model_architecture.json', 'w') as f:
    f.write(model.to_json())