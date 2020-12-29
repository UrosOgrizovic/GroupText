import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, GRU, Bidirectional, Embedding, MaxPooling1D, InputLayer, Flatten, Reshape, Lambda
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import Input, Model
import numpy as np
import math


def create_model(word_index_len, embedding_matrix, embedding_dim, num_sen_per_doc, sen_len):
    model = Sequential()
    embedding_layer = Embedding(word_index_len + 1, embedding_dim, mask_zero=True,
                                input_shape=(num_sen_per_doc, sen_len), trainable=True, weights=[embedding_matrix])
    model.add(embedding_layer)
    model.add(Lambda(lambda x: x, output_shape=lambda s: s))
    model.add(
        Reshape((num_sen_per_doc, sen_len * embedding_dim), input_shape=(num_sen_per_doc * sen_len, embedding_dim)))
    model.add(Bidirectional(GRU(16, return_sequences=True, dropout=0.2, activation='relu')))
    model.add(Bidirectional(GRU(16, return_sequences=True, activation='relu')))
    # model.add(MaxPooling1D())

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # categories: start-of-segment (SOS), not SOS
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=["accuracy"])
    return model


def custom_generator(x, y, batch_size):
    idx = 0
    while True:
        yield x[idx * batch_size:(idx + 1) * batch_size], y[idx * batch_size:(idx + 1) * batch_size]
        if idx == len(x) // batch_size:   # reset idx at end of epoch
            idx = 0
        else:
            idx += 1


def train_model(model, save_model_path, x_train, y_train, batch_size, epochs, validation_split):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4,
                                       mode='min')
    print(x_train.shape)

    x_val = x_train[-math.ceil(validation_split * len(x_train)):]
    x_train = x_train[:-math.ceil(validation_split * len(x_train))]
    y_val = y_train[-math.ceil(validation_split * len(y_train)):]
    y_train = y_train[:-math.ceil(validation_split * len(y_train))]
    print(x_train.shape)
    print(len(x_train) // batch_size)
    tr_gen = custom_generator(x_train, y_train, batch_size)
    val_gen = custom_generator(x_val, y_val, batch_size)
    # train_generator = TimeseriesGenerator(x_train, y_train, length=5, batch_size=batch_size)
    # val_generator = TimeseriesGenerator(x_val, y_val, length=5, batch_size=batch_size)
    history = model.fit_generator(generator=tr_gen,
                                  steps_per_epoch=len(x_train) // batch_size,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=len(x_val) // batch_size,
                                  callbacks=[early_stopping, mcp_save, reduce_lr_loss],
                                  verbose=2)
    # history = model.fit_generator(train_generator,
    #                     steps_per_epoch=len(x_train) // batch_size,
    #                     epochs=epochs,
    #                     validation_data=val_generator,
    #                     validation_steps=int(len(x_train)*validation_split) // batch_size,
    #                     callbacks=[early_stopping, mcp_save, reduce_lr_loss],
    #                     verbose=2)
    return history
