import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from keras import Sequential
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.layers import Dense, GRU, Bidirectional, Embedding
# from tensorflow.keras.layers import (
#     Layer,
#     Dense,
#     Input,
#     GRU,
#     Bidirectional,
#     MaxPooling1D,
#     Flatten,
#     Embedding
# )


def create_model(word_index_len, embedding_matrix, embedding_dim):
    # model = tf.keras.Sequential()
    model = Sequential()
    embedding_layer = Embedding(word_index_len + 1, embedding_dim)
    # see https://github.com/keras-team/keras/issues/4753#issuecomment-629618860
    embedding_layer.build((None,))
    # weights in Embedding layer is deprecated, using layer.set_weights instead
    embedding_layer.set_weights([embedding_matrix])
    embedding_layer.trainable = False
    model.add(embedding_layer)
    model.add(Bidirectional(GRU(256, return_sequences=True, dropout=0.2, activation='relu')))
    model.add(Bidirectional(GRU(256, activation='relu')))
    model.add(Dense(2, activation="softmax"))  # categories: start-of-segment (SOS), not SOS
    model.compile(loss=CategoricalCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(lr=0.001),
                  metrics=["accuracy"])
    return model
