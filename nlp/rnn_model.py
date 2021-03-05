import tensorflow as tf
from keras import Sequential, Input, Model, initializers
from keras.optimizers import RMSprop
from keras.layers import Dense, GRU, Bidirectional, Embedding,\
    Reshape, Lambda, Dropout, BatchNormalization, Activation,\
    GlobalMaxPooling1D, Masking
from keras.metrics import Accuracy, MeanSquaredError
from keras.preprocessing.sequence import pad_sequences
import helpers
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import data_operations
import helpers
import os
import gc
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # run on CPU

tf.random.set_seed(3)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

my_init = initializers.glorot_uniform(seed=1)


def add_bidirectional(model, return_sequences=True):
    model.add(Bidirectional(GRU(128, return_sequences=return_sequences,
                            kernel_initializer=my_init)))
    return model


def add_dense(model):
    model.add(Dense(32, kernel_initializer=my_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    return model


def custom_loss(y_true, y_pred):
    # "A coefficient to use on the positive examples."
    pos_weight = 5.0 / 1.0
    # make both y_true and y_pred float32
    y_true = tf.cast(y_true, tf.float32)

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true,
                                                                   y_pred,
                                                                   pos_weight))
    return loss


def custom_generator(num_docs, batch_size, set='tr'):
    path = helpers.get_path(num_docs, set)
    counter = 0
    curr_pos = 0
    num_batches = num_docs // batch_size
    while True:
        x_batch, y_batch, curr_pos = data_operations.read_dumps_in_batches(path, batch_size, curr_pos)
        avg_doc_len = helpers.find_avg_doc_length_in_batch(x_batch)
        x_batch = pad_sequences(x_batch, avg_doc_len, padding='post', truncating='post', dtype=np.float32)    # (batch_size, doc_len, sen_len)
        y_batch = pad_sequences(y_batch, avg_doc_len, padding='post', truncating='post', dtype=np.float32)    # (batch_size, doc_len)
        print(f'AVG DOC LEN: {avg_doc_len}')
        counter += 1

        # converting from np.array to tensor because of https://github.com/tensorflow/tensorflow/issues/44705#issuecomment-725803328
        yield tf.convert_to_tensor(x_batch), tf.convert_to_tensor(np.reshape(y_batch, (y_batch.shape[0], y_batch.shape[1], 1)))

        # restart counter to yield data in the next epoch as well
        if counter >= num_batches:
            counter = 0
            curr_pos = 0


def train_model_generator(batch_size, sent_enc, model, num_epochs, save_model_path, num_docs):

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min',
                                   min_delta=0.1)
    mcp_save = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss',
                               mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                       verbose=1, min_delta=0.05, mode='min')
    # first 0.8 for tr/tst split, second 0.8 for tr/val split
    len_tr = num_docs * 0.8 * 0.8
    len_val = num_docs * 0.8 * 0.2
    num_batches_tr = int(len_tr//batch_size)
    num_batches_val = int(len_val//(batch_size))

    custom_model = CustomModel(Encoder(batch_size), batch_size)
    custom_model.compile(optimizer=RMSprop(lr=1e-5),
                         loss=custom_loss)

    return custom_model.fit(custom_generator(num_docs, batch_size), epochs=num_epochs,
                            steps_per_epoch=num_batches_tr,
                            validation_data=custom_generator(num_docs, batch_size, 'val'),
                            validation_steps=num_batches_val,
                            callbacks=[early_stopping, mcp_save, reduce_lr])


class Encoder(Model):
    def __init__(self, batch_size, sen_len=20):
        super(Encoder, self).__init__()
        self.masking = Masking(mask_value=0., input_shape=(sen_len, 1))
        # why batch_input_shape instead of input_shape: https://github.com/tensorflow/tensorflow/issues/37942#issuecomment-604981240
        self.bidir_first_lower = Bidirectional(GRU(128, return_sequences=True,
                            kernel_initializer=my_init),
                            # batch_input_shape=(batch_size, sen_len, 1)
                            )
        self.bidir = Bidirectional(GRU(128, return_sequences=True,
                            kernel_initializer=my_init))
        self.pooling = GlobalMaxPooling1D()

    def call(self, x):
        x = self.masking(x)
        # print(self.masking.input_mask)
        # print(self.masking.output_mask)
        x = self.bidir_first_lower(x)
        x = self.bidir(x)
        x = self.pooling(x)
        return x


class CustomModel(Model):
    def __init__(self, encoder, batch_size, sen_len=20):
        super(CustomModel, self).__init__()
        self.metric = MeanSquaredError()

        self.encoder = encoder
        # why batch_input_shape instead of input_shape: https://github.com/tensorflow/tensorflow/issues/37942#issuecomment-604981240
        self.bidir_first_upper = Bidirectional(GRU(128, return_sequences=True,
                            kernel_initializer=my_init))
                            # batch_input_shape=(batch_size, 1, 256))
        self.bidir = Bidirectional(GRU(128, return_sequences=True,
                            kernel_initializer=my_init))
        self.bidir_last = Bidirectional(GRU(128,
                            kernel_initializer=my_init))
        self.dense = Dense(1, activation="sigmoid")
        self.batch_size = batch_size

    def compile(self, optimizer, loss):
        super().compile()
        self.run_eagerly = True
        self.optimizer = optimizer
        self.loss = loss

    def call(self, x):  # https://github.com/tensorflow/tensorflow/issues/43173
        # x.shape = (batch_size, doc_len, sen_len)
        # x.shape = (batch_size * doc_len, 1, 256)
        print(f'Shape of input {tf.shape(x)}')
        out = self.bidir_first_upper(x)
        out = self.bidir(out)
        out = self.bidir_last(out)
        # probability that current sentence starts a new segment
        out = self.dense(out)
        return out

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            doc_len = tf.shape(x)[1].numpy() # all docs in batch have same len
            sen_len = tf.shape(x)[2].numpy() # all sentences in corpus have same len

            encodings = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                        element_shape=tf.TensorShape([doc_len, 1, 256]))

            for doc_idx in range(len(x)):
                curr_doc = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                        element_shape=tf.TensorShape([1, 256]))
                for sen_idx in range(len(x[doc_idx])):
                    sen = tf.expand_dims(x[doc_idx][sen_idx], -1)   # (20,) -> (20, 1)
                    # tf.shape is necessary - https://github.com/tensorflow/models/issues/6245#issuecomment-623877638
                    sen = tf.reshape(sen, (1, tf.shape(sen)[0], tf.shape(sen)[1]))  # (20, 1) -> (1, 20, 1)
                    orig_sen = sen
                    print(f'Suma pre {np.sum(sen)}')
                    sen = self.encoder(sen)
                    print(f'Suma posle {np.sum(sen)}')
                    if math.isnan(np.sum(sen)):
                        print(f'Orig sen {orig_sen}')
                        print(f'Sen {sen}')
                        exit()
                    curr_doc = curr_doc.write(sen_idx, sen)

                encodings = encodings.write(doc_idx, tf.expand_dims(curr_doc.concat(), 1))
            encodings = encodings.concat()
            assert all(tf.shape(encodings) == [self.batch_size * doc_len, 1, 256])

            # calculate predictions
            y_pred = self(encodings, training=True)

            # the next reshape is so that y and y_pred have the same shape
            y_pred = tf.reshape(y_pred, (self.batch_size, tf.shape(y_pred)[0] / self.batch_size, 1))

            loss = self.loss(y, y_pred)

        # Gradients
        training_vars = self.trainable_variables + self.encoder.trainable_variables

        gradients = tape.gradient(loss, training_vars)
        # print(gradients)
        print('------------------------------------------')
        # print(loss)
        if np.isnan(loss.numpy()):
            print('U NAN-U SMO')
            encoder_weights = []
            for layer in self.encoder.layers:
                encoder_weights.append(layer.get_weights())
            with open("encoder_weights.txt", "w") as output:
                output.write(str(encoder_weights))
            with open("bidir_first_upper_weights.txt", "w") as output:
                output.write(str(self.bidir_first_upper.get_weights()))
            with open("bidir_weights.txt", "w") as output:
                output.write(str(self.bidir.get_weights()))
            with open("bidir_last_weights.txt", "w") as output:
                output.write(str(self.bidir_last.get_weights()))
            with open("dense_weights.txt", "w") as output:
                output.write(str(self.dense.get_weights()))
            # print(f'x {x}')
            # for i in range(len(y)):
            #     for j in range(len(y[i])):
            #         print(y[i][j], y_pred[i][j])
            exit()

        # Step with optimizer
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        self.metric.update_state(y, y_pred)

        return {"loss": loss, "MSE": self.metric.result()}

    # def test_step(self, data):
    #     x, y = data

    #     encodings = encode_sentences(x, self.sen_enc)

    #     # Compute predictions
    #     y_pred = self.classifier(encodings, training=True)

    #     # Updates the metrics tracking the loss
    #     loss = self.loss(y, y_pred)

    #     # Update the metrics.
    #     self.metric.update_state(y, y_pred)
    #     return {"loss": loss, "MSE": self.metric.result()}


if __name__ == '__main__':
    num_docs = 10000
    num_epochs = 1
    batch_size = 4
    train_model_generator(batch_size, None, None, num_epochs, f'saved_models/model{num_docs}.h5', num_docs)
    # generator = custom_generator(None, num_docs, 32)
    # idx = 0
    # while idx < 310:
    #     idx += 1
    #     x, y = next(generator)
    #     # print(tf.shape(x), tf.equal(tf.shape(x), (32, 55 ,20)))
    #     # print(tf.shape(y))
    #     for doc in x:
    #         for sen in doc:
    #             if all(tf.shape(sen) != (20,)):
    #                 print(f'AAAAA {tf.shape(sen)}')

