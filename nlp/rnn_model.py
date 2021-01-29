import tensorflow as tf
from keras import Sequential
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Dense, GRU, Bidirectional, Embedding,\
    Reshape, Lambda, Dropout, BatchNormalization, Activation
from keras.preprocessing.sequence import pad_sequences
import helpers
from keras.regularizers import l2
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import data_operations
import helpers

tf.random.set_seed(3)  # Tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

my_init = initializers.glorot_uniform(seed=1)
dropout_amount = 0.1


def add_bidirectional(model):
    model.add(Bidirectional(GRU(128, return_sequences=True,
                            kernel_initializer=my_init)))
    model.add(BatchNormalization())     # add BN before non-linearity
    model.add(Activation('relu'))
    return model


def add_dense(model):
    model.add(Dense(32, kernel_initializer=my_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    return model


@tf.function()
def custom_loss(y_true, y_pred):
    # "A coefficient to use on the positive examples."
    pos_weight = 5.0 / 1.0
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true,
                                                                   y_pred,
                                                                   pos_weight))
    return loss


def create_model(word_index_len, embedding_matrix, embedding_dim, sen_len=20):
    model = Sequential()
    # embedding_layer = Embedding(input_dim=word_index_len + 1,
    #                             output_dim=embedding_dim,
    #                             mask_zero=True,
    #                             input_shape=(None, sen_len),
    #                             # trainable=True, weights=[embedding_matrix])
    #                             trainable=True)
    # model.add(embedding_layer)
    # model.add(Lambda(lambda x: x, output_shape=lambda s: s))

    # # -1 stands for shape inference
    # model.add(Reshape((-1, sen_len * embedding_dim)))

    model.add(Bidirectional(GRU(128, return_sequences=True,
                            kernel_initializer=my_init),
                            input_shape=(None, sen_len)))
    model.add(BatchNormalization())     # add BN before non-linearity
    model.add(Activation('relu'))

    model = add_bidirectional(model)
    model = add_bidirectional(model)
    model = add_bidirectional(model)

    # model = add_dense(model)

    # output probability that current sentence starts a new segment
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss=custom_loss,
                  optimizer=Adam(lr=1e-6),
                  metrics=["accuracy"])
    return model


def custom_generator(num_docs, num_batches, batch_size, set='tr'):
    path = helpers.get_path(num_docs, set)
    counter = 0
    curr_pos = 0
    while True:
        x_batch, y_batch, curr_pos = data_operations.read_dumps_in_batches(path, batch_size, curr_pos)
        avg_doc_len = helpers.find_avg_doc_length_in_batch(x_batch)
        x_batch = pad_sequences(x_batch, avg_doc_len, padding='post', truncating='post')
        y_batch = pad_sequences(y_batch, avg_doc_len, padding='post', truncating='post')

        counter += 1
        yield x_batch, np.reshape(y_batch, (y_batch.shape[0], y_batch.shape[1], 1))

        # restart counter to yield data in the next epoch as well
        if counter >= num_batches:
            counter = 0
            curr_pos = 0


def lr_decay(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0005
    else:
        return 0.0001


def train_model_generator(batch_size, model, num_epochs, save_model_path, num_docs):

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='min',
                                   min_delta=0.1)
    mcp_save = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss',
                               mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                       verbose=1, min_delta=0.05, mode='min')
    # first 0.8 for tr/tst split, second 0.8 for tr/val split
    len_tr = num_docs * 0.8 * 0.8
    len_val = num_docs * 0.8 * 0.2
    num_batches_tr = int(len_tr//batch_size)
    num_batches_val = int(len_val//(batch_size*2))

    return model.fit_generator(custom_generator(num_docs, num_batches_tr, batch_size), epochs=num_epochs,
                               steps_per_epoch = num_batches_tr,
                               validation_data = custom_generator(num_docs, num_batches_val, batch_size*2, 'val'),
                               validation_steps = num_batches_val,
                            #    callbacks=[LearningRateScheduler(lr_decay, verbose=1)])
                               callbacks=[early_stopping, mcp_save, reduce_lr])
