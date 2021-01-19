
from keras import Sequential
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Dense, GRU, Bidirectional, Embedding,\
    Reshape, Lambda, Dropout, BatchNormalization, Activation
import tensorflow as tf
# from keras.regularizers import l2

tf.random.set_seed(3)  # Tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

my_init = initializers.glorot_uniform(seed=1)
dropout_amount = 0.2


def add_bidirectional(model):
    model.add(Bidirectional(GRU(64, return_sequences=True,
                            kernel_initializer=my_init)))
    model.add(BatchNormalization())     # add BN before non-linearity
    model.add(Activation('relu'))
    # model.add(Dropout(dropout_amount))     # mask the outputs
    return model


def add_dense(model):
    # no need for bias before BN, because BN normalizes to mean=0 and var=1
    model.add(Dense(32, use_bias=False, kernel_initializer=my_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_amount))
    return model


@tf.function()
def custom_loss(y_true, y_pred):
    # "A coefficient to use on the positive examples."
    pos_weight = 10.0 / 1.0
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true,
                                                                   y_pred,
                                                                   pos_weight))
    return loss


def create_model(word_index_len, embedding_matrix, embedding_dim,
                 num_sen_per_doc, sen_len):
    # TODO: odradi fit_generator
    model = Sequential()
    embedding_layer = Embedding(word_index_len + 1, embedding_dim,
                                mask_zero=True,
                                input_shape=(num_sen_per_doc, sen_len),
                                trainable=True, weights=[embedding_matrix])
    model.add(embedding_layer)
    model.add(Lambda(lambda x: x, output_shape=lambda s: s))
    model.add(Reshape((num_sen_per_doc, sen_len * embedding_dim),
              input_shape=(num_sen_per_doc * sen_len, embedding_dim)))

    model = add_bidirectional(model)
    model = add_bidirectional(model)
    model = add_bidirectional(model)

    model = add_dense(model)

    # output probability that current sentence starts a new segment
    model.add(Dense(1, activation="sigmoid"))
    # model.compile(loss='binary_crossentropy',
    model.compile(loss=custom_loss,
                  optimizer=Adam(lr=1e-3),
                  metrics=["accuracy"])
    return model
