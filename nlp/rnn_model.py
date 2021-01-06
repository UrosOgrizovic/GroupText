
from keras import Sequential
from keras.optimizers import Nadam
from keras.layers import Dense, GRU, Bidirectional, Embedding,\
    Reshape, Lambda, Dropout, BatchNormalization, Activation
import tensorflow as tf
from keras.regularizers import l2

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def create_model(word_index_len, embedding_matrix, embedding_dim,
                 num_sen_per_doc, sen_len):
    model = Sequential()
    embedding_layer = Embedding(word_index_len + 1, embedding_dim,
                                mask_zero=True,
                                input_shape=(num_sen_per_doc, sen_len),
                                trainable=True, weights=[embedding_matrix])
    model.add(embedding_layer)
    model.add(Lambda(lambda x: x, output_shape=lambda s: s))
    model.add(Reshape((num_sen_per_doc, sen_len * embedding_dim),
              input_shape=(num_sen_per_doc * sen_len, embedding_dim)))
    ''' 1. the dropout parameter masks the inputs
    2. the recurrent_dropout parameter masks the connections
        between the hidden units
    '''
    model.add(Bidirectional(GRU(16, return_sequences=True,
                                kernel_regularizer=l2(1e-4), dropout=0.2,
                                recurrent_regularizer=l2(1e-4),
                                recurrent_dropout=0.2)))
    model.add(BatchNormalization())     # add BN before non-linearity
    model.add(Activation('relu'))
    model.add(Dropout(0.2))     # mask the outputs
    # model.add(Bidirectional(GRU(16, return_sequences=True, dropout=0.5,
    #                             activation='relu')))
    # model.add(MaxPooling1D())

    # no need for bias before BN, because BN normalizes to mean=0 and var=1
    model.add(Dense(32, kernel_regularizer=l2(1e-4), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # output probability that current sentence starts a new segment
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=1e-4, clipvalue=0.5),
                  metrics=["accuracy"])
    return model
