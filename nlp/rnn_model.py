import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from keras import Sequential
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.layers import Dense, GRU, Bidirectional, Embedding, MaxPooling1D
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
    embedding_layer = Embedding(word_index_len + 1, embedding_dim, mask_zero=True)
    # see https://github.com/keras-team/keras/issues/4753#issuecomment-629618860
    embedding_layer.build((None,))
    # weights in Embedding layer is deprecated, using layer.set_weights instead
    embedding_layer.set_weights([embedding_matrix])
    embedding_layer.trainable = False
    model.add(embedding_layer)
    model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.2, activation='relu')))
    # model.add(GRU(128, return_sequences=True, dropout=0.2, activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.2, activation='relu')))
    # model.add(MaxPooling1D())
    # model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.2, activation='relu')))
    # model.add(MaxPooling1D())
    # model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.2, activation='relu')))
    # model.add(MaxPooling1D())
    # model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.2, activation='relu')))
    # model.add(MaxPooling1D())
    model.add(Bidirectional(GRU(128, activation='relu')))
    # model.add(GRU(64, return_sequences=True, dropout=0.2, activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(GRU(64, activation='relu'))
    # model.add(Dense(2, activation="softmax"))  # categories: start-of-segment (SOS), not SOS
    model.add(Dense(1, activation="sigmoid"))  # categories: start-of-segment (SOS), not SOS
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=["accuracy"])
    return model


def train_model(model, save_model_path, x_train, y_train, batch_size, epochs, validation_split):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4,
                                       mode='min')
    history = model.fit(x=x_train, y=y_train, validation_split=validation_split, steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs,
                        validation_steps=len(x_train)*validation_split // batch_size,
                        callbacks=[early_stopping, mcp_save, reduce_lr_loss],
                        verbose=2)
    return history
