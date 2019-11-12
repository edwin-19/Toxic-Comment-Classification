from __future__ import absolute_import, division

import tensorflow as tf 
from keras.layers import Dense, Input, Embedding
from keras.layers import Lambda, Dropout, Activation
from keras.layers import SpatialDropout1D, Reshape, GlobalAveragePooling1D
from keras.layers import merge, Flatten, Bidirectional
from keras.layers import Flatten, Bidirectional, CuDNNGRU
from keras.layers import add, Conv1D, GlobalMaxPooling1D

from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import BatchNormalization
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
from keras.optimizers import Adam

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps
    Uses 1 paramater pr .channel to compute the attention value for a single timestep
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)
    
    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3
        
        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)
    
    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [
                (input_shape[0], output_len), 
                (input_shape[0], input_shape[1])
            ]

        return (input_shape[0], output_len)

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
    

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] * self.k)

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        
        return Flatten()(top_k)

def get_kmax_text_cnn(nb_words, embedding_dim, embedding_matrix, max_squence_length, out_size):
    embedding_layer = Embedding(
        nb_words, embedding_dim, weights=[embedding_matrix],
        input_length=max_squence_length, trainable=False
    )
    
    filter_nums = 180
    drop = 0.6
    
    comment_input = Input(shape=(max_squence_length,), dtype='int32')
    embeded_sequences = embedding_layer(comment_input)
    embeded_sequences = SpatialDropout1D(0.2)(embeded_sequences)
    
    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    maxpool_0 = KMaxPooling(k=3)(conv_0)
    maxpool_1 = KMaxPooling(k=3)(conv_1)
    maxpool_2 = KMaxPooling(k=3)(conv_2)
    maxpool_3 = KMaxPooling(k=3)(conv_3)
    
    merged_tensor = merge(
        [maxpool_0, maxpool_1, maxpool_2, maxpool_3],
        mode='concat', concat_axis=1
    )
    output = Dropout(drop)(merged_tensor)
    output = Dense(units=144, activation='relu')(output)
    output = Dense(units=out_size, activation='sigmoid')(output)
    
    model = Model(inputs=comment_input, outputs=output)
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-7)
    )
    
    print(model.summary())
    return model

def get_av_rnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    recurrent_units = 60
    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(
        nb_words, embedding_dim,
        weights=[embedding_matrix], input_length=max_sequence_length,
        trainable=False
    )(input_layer)
    embedding_layer = SpatialDropout1D(0.25)(embedding_layer)
    
    rnn_1 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    rnn_2 = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(rnn_1)
    x = concatenate([rnn_1, rnn_2], axis=2)
    
    last = Lambda(lambda t: t[:, -1], name='last')(x)
    maxpool = GlobalMaxPooling1D()(x)
    attn = AttentionWeightedAverage()(x)
    average = GlobalAveragePooling1D()(x)
    
    all_views = concatenate([last, maxpool, average, attn], axis=1)
    x = Dropout(0.5)(all_views)
    x = Dense(144, activation='relu')(x)
    output_layer = Dense(out_size, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-6, clipvalue=5))
    print(model.summary())
    return model