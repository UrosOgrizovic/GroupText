import rnn_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import constants
import numpy as np
import os
# import random
from text_preprocessing import split_text_into_sentences, parse_sentence
from tensorflow.keras.models import load_model
import data_operations
import helpers
import pickle
# import re
from sys import getsizeof

np.random.seed(3)


def train_model(batch_size, num_epochs, validation_split,
        save_model_path, num_docs):
    print("Creating sentence encoder...")
    sent_enc = rnn_model.sentence_encoder()
    print("Finished creating sentence encoder")

    print("Creating model...")
    upper_net = rnn_model.upper_network()
    print("Finished creating model")
    print(upper_net.summary())
    history = rnn_model.train_model_generator(batch_size, sent_enc, upper_net, num_epochs,
                                              save_model_path, num_docs)
    data_operations.plot_training_and_validation_data(history,
                                                      helpers.abbreviate_num_to_str(num_docs))
    model.save(save_model_path)
    print(f"Finished training and saved model to path: {save_model_path}")
    return model


def evaluate_model(text, tokenizer, sen_pad_len, doc_pad_len, model):
    text = split_text_into_sentences(text)
    for i in range(len(text)):
        text[i] = parse_sentence(text[i])

    # save text as string for printing
    text_str = text

    seq_text = tokenizer.texts_to_sequences(text)
    seq_text = pad_sequences(seq_text, sen_pad_len, padding='post',
                             truncating='post')
    seq_text = seq_text.reshape(1, seq_text.shape[0], seq_text.shape[1])
    len_before_padding = seq_text.shape[1]
    seq_text = pad_sequences(seq_text, doc_pad_len, padding='post',
                             truncating='post')

    predictions = model.predict(seq_text)[0]
    # not interested in the predictions for the padded elements
    predictions = predictions[-len_before_padding:]
    for i in range(len(predictions)):
        print(f"{predictions[i][0]}: {text_str[i]:50}")


if __name__ == "__main__":
    num_docs = 100000
    num_in_path = helpers.abbreviate_num_to_str(num_docs)
    save_model_path = "saved_models/model" + str(num_docs) + ".h5"
    read_docs_path = "extracted/wiki_727K"
    num_words_to_keep = 10000
    tokenizer_path = f'tokenizer_{num_in_path}.pkl'
    batch_size, num_epochs, validation_split, test_split = 32, 30, 0.2, 0.8
    sentence_document_mapping = {}  # sentence_index: document_index
    doc_batch_size = 100
    document_index, sentence_index = -1, 0
    y_tr, y_tst = [], []
    list_of_all_words = []
    curr_pos = 0
    for num_docs_loaded in range(doc_batch_size, num_docs+1, doc_batch_size):
        y_tr, y_tst, list_of_all_words, sentence_document_mapping, curr_pos,\
        sentence_index, document_index = \
            data_operations.read_docs_in_batches(read_docs_path, doc_batch_size, curr_pos, num_in_path,
            sentence_document_mapping, list_of_all_words, test_split, y_tr, y_tst, sentence_index,
            document_index, num_docs, num_docs_loaded)
    tokenizer = data_operations.get_tokenizer(tokenizer_path, list_of_all_words, num_words_to_keep)
    x_tr, y_tr, x_val, y_val, x_tst, y_tst = data_operations.process_loaded_docs(y_tr, y_tst,
                                                                                list_of_all_words,
                                                                                sentence_document_mapping,
                                                                                constants.AVG_SEN_LEN,
                                                                                tokenizer, num_in_path)
    exit()

    tokenizer_path = f'tokenizer_100k.pkl'
    # tokenizer = data_operations.get_tokenizer(tokenizer_path, list_of_all_words)

    print(f'Tokenizer size: {getsizeof(tokenizer)} bytes')

    texts = [constants.TEXT_1, constants.TEXT_2,
             constants.TEXT_3, constants.TEXT_4, constants.TEXT_5]
    train = True
    model = None
    if train:
        model = train_model(batch_size, num_epochs, validation_split,
            save_model_path, num_docs)
        # x_tst, y_tst = data_operations.load_x_y(f'data/dump_tst_{num_in_path}.pkl')
        # x_tst = np.expand_dims(x_tst, axis=0)
        # y_tst = np.expand_dims(y_tst, axis=0)
        # print(x_tst.shape, y_tst.shape)
        # print(y_tst)
        # print(f"Evaluating model: [loss, accuracy]: {model.evaluate(x_tst, y_tst)}")
    else:
        ''' if loss is required (e.g. for model.evaluate()), use
        model = load_model(save_model_path,
        custom_objects={'custom_loss': rnn_model.custom_loss(y_true, y_pred)})
        if the model will only be used for inference, compile=False is your friend :)
        '''
        # compile=False added because of custom loss function
        model = load_model(save_model_path, compile=False)


    # num_docs_tr = x_tr.shape[0]
    # num_sen_per_doc = len(x_tr[0])
    # y_tr = y_tr.reshape(num_docs_tr, num_sen_per_doc, 1)

    # seq_text = x_tr[0].reshape(1, x_tr[0].shape[0], x_tr[0].shape[1])
    # label = y_tr[0].reshape(1, y_tr[0].shape[0], y_tr[0].shape[1])
    # predictions = model.predict(seq_text)
    # print(predictions)
    # for i in range(len(predictions[0])):
    #     print(f"{predictions[0][i][0]}: {y_tr[0][i]}")
    # exit()

    # evaluate_model(texts[4], tokenizer, sen_pad_len,
    #                doc_pad_len, model)
