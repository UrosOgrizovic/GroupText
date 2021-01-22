
from keras import Sequential
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Dense, GRU, Bidirectional, Embedding,\
    Reshape, Lambda, Dropout, BatchNormalization, Activation
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from helpers import find_lengths_in_batch
# from keras.regularizers import l2
import numpy as np

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
    # print(y_true.shape, y_true)
    # print(y_pred.shape, y_pred)
    # "A coefficient to use on the positive examples."
    pos_weight = 5.0 / 1.0
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true,
                                                                   y_pred,
                                                                   pos_weight))
    return loss


def create_model(word_index_len, embedding_matrix, embedding_dim,
                 num_sen_per_doc, sen_len):
    model = Sequential()
    embedding_layer = Embedding(input_dim=word_index_len + 1,
                                output_dim=embedding_dim,
                                mask_zero=True,
                                input_shape=(None, sen_len),
                                trainable=True, weights=[embedding_matrix])
    model.add(embedding_layer)
    model.add(Lambda(lambda x: x, output_shape=lambda s: s))

    # -1 stands for shape inference
    model.add(Reshape((-1, sen_len * embedding_dim)))

    model = add_bidirectional(model)
    model = add_bidirectional(model)
    model = add_bidirectional(model)

    model = add_dense(model)

    # output probability that current sentence starts a new segment
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss=custom_loss,
                  optimizer=Adam(lr=1e-3),
                  metrics=["accuracy"])
    return model


def custom_generator(x, y, batch_size):
    samples_per_epoch = len(x)
    number_of_batches = samples_per_epoch/batch_size
    counter = 0
    while True:
        x_batch = x[batch_size*counter:batch_size*(counter+1)]
        y_batch = y[batch_size*counter:batch_size*(counter+1)]
        avg_doc_len, _ = find_lengths_in_batch(x_batch)
        x_batch = pad_sequences(x_batch, avg_doc_len, padding='post', truncating='post')
        y_batch = pad_sequences(y_batch, avg_doc_len, padding='post', truncating='post')

        # print(y_batch.shape)
        # print("gotovo pedovanje")
        counter += 1
        yield x_batch, np.reshape(y_batch, (y_batch.shape[0], y_batch.shape[1], 1))

        # restart counter to yield data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0


def train_model_generator(x, y, batch_size, model, num_epochs):
    length = int(0.8*len(x))
    x_tr = x[:length]
    y_tr = y[:length]
    x_val = x[length:]
    y_val = y[length:]
    # print(len(x_tr), y_tr.shape, y_tr[0].shape)
    # print(y_tr)
    # exit()
    # print(x_tr[0], type(x_tr[0]))
    # x_tr = pad_sequences(x_tr, 22, padding='post', truncating='post')
    # print(x_tr[0], type(x_tr[0]))
    # exit()
    return model.fit_generator(custom_generator(x_tr, y_tr, batch_size), epochs=num_epochs,
                               steps_per_epoch = len(x_tr)//batch_size,
                               validation_data = custom_generator(x_val, y_val, batch_size*2),
                               validation_steps = len(x_val)//(batch_size*2))
