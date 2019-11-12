import numpy as np 

def load_embedding(embedding_path):
    embeddings_index = {}
    f = open(embedding_path, 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        try:
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        except:
            print("Err on ", values[:2])

    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index