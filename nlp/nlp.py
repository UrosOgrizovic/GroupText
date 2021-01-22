import rnn_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import constants
import numpy as np
import os
import gc
# import random
from text_preprocessing import split_text_into_sentences, parse_sentence
from tensorflow.keras.models import load_model
import data_operations
import helpers
# import re

np.random.seed(3)
SEGMENT_DELIMITER = "========"


def calculate_embedding_matrix(word_index, embedding_dim):
    embeddings_index = {}
    f = open(os.path.join("glove", 'glove.6B.' +
                          str(embedding_dim) + 'd.txt'), encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def train_model(
        x_tr, y_tr, batch_size, num_epochs, validation_split, save_model_path,
        tokenizer, embedding_matrix, embedding_dim, num_sen_per_doc, sen_len,
        num_docs):
    print("Creating model...")
    model = rnn_model.create_model(
        len(tokenizer.word_index),
        embedding_matrix, embedding_dim, num_sen_per_doc, sen_len)
    print("Finished creating model")
    print(model.summary())
    history = rnn_model.train_model_generator(x_tr, y_tr, batch_size, model, num_epochs,
                                              save_model_path)
    data_operations.plot_training_and_validation_data(history,
                                                      helpers.abbreviate_num_to_str(num_docs))
    model.save(save_model_path)
    print(f"Finished training and saved model to path: {save_model_path}")
    return model


def evaluate_model(text, tokenizer, sen_pad_len,
                   doc_pad_len, model):

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


def get_percentage_of_segment_starting_sentences(y, doc_pad_len):
    num_ones_tr = 0
    for doc in y:
        for i in range(doc_pad_len):
            if doc[i] == 1:
                num_ones_tr += 1
    total_num_sen_y_tr = doc_pad_len * len(y_tr)
    ratio = total_num_sen_y_tr / num_ones_tr
    percentage = round(1 / ratio, 2) * 100
    # print(f"Total number of sentences in y_tr {total_num_sen_y_tr}")
    # print(f"Number of segment-starting sentences in y_tr {num_ones_tr}")
    # print(f"Non-segment starting sentences/segment-starting sentences {ratio}")
    return percentage


if __name__ == "__main__":
    num_docs = 1000
    save_model_path = "saved_models/model" + str(num_docs) + ".h5"
    read_docs_path = "extracted/wiki_727K"
    y_tr, y_tst, list_of_all_words, sentence_document_mapping, avg_sen_len = \
        data_operations.read_docs(read_docs_path, num_docs)
    # convert strings to numbers
    tokenizer = Tokenizer()
    print("Fitting tokenizer...")
    tokenizer.fit_on_texts(list_of_all_words)
    print("Finished fitting tokenizer")

    sequences = tokenizer.texts_to_sequences(list_of_all_words)
    del list_of_all_words
    gc.collect()

    sen_pad_len = avg_sen_len + 5

    # perform padding/truncating (truncate ends of sentences, because
    # beginnings are more important for this task)
    sequences = pad_sequences(sequences, sen_pad_len, padding='post',
                              truncating='post')
    # 80/20 train-test split
    x_tr, x_tst = sequences[:len(y_tr)], sequences[len(y_tr):]

    del sequences
    gc.collect()

    # group training data as documents
    documents_x_tr, documents_y_tr = [], []
    documents_x_tst, documents_y_tst = [], []
    curr_doc_x, curr_doc_y = [x_tr[0]], [y_tr[0]]
    for i in range(1, len(x_tr)):
        # new document
        if sentence_document_mapping[i] != sentence_document_mapping[i-1]:
            documents_x_tr.append(curr_doc_x)
            documents_y_tr.append(np.array(curr_doc_y, dtype='object'))
            curr_doc_x = [x_tr[i]]
            curr_doc_y = [y_tr[i]]
        else:
            curr_doc_x.append(x_tr[i])
            curr_doc_y.append(y_tr[i])

    # group test data as documents
    curr_doc_x, curr_doc_y = [x_tst[0]], [y_tst[0]]
    for i in range(1, len(x_tst)):
        # new document
        if sentence_document_mapping[i + len(x_tr)-1] != \
                sentence_document_mapping[i-1+len(x_tr)-1]:
            documents_x_tst.append(curr_doc_x)
            documents_y_tst.append(np.array(curr_doc_y, dtype='object'))
            curr_doc_x = [x_tst[i]]
            curr_doc_y = [y_tst[i]]
        else:
            curr_doc_x.append(x_tst[i])
            curr_doc_y.append(y_tst[i])

    del sentence_document_mapping
    gc.collect()

    documents_x_tr[-1].append(x_tr[-1])
    documents_x_tst[-1].append(x_tst[-1])
    np.append(documents_y_tr[-1], y_tr[-1])
    np.append(documents_y_tst, y_tst[-1])

    documents_x_tr = np.array([np.array(lst, dtype='object')
                               for lst in documents_x_tr], dtype='object')
    documents_x_tst = np.array([np.array(lst, dtype='object')
                                for lst in documents_x_tst], dtype='object')

    x_tr = np.array(documents_x_tr, dtype='object')
    y_tr = np.array(documents_y_tr, dtype='object')
    x_tst = np.array(documents_x_tst, dtype='object')
    y_tst = np.array(documents_y_tst, dtype='object')

    del documents_x_tr
    del documents_y_tr
    del documents_x_tst
    del documents_y_tst
    del curr_doc_x
    del curr_doc_y
    gc.collect()

    doc_lens = []
    for doc in x_tr:
        doc_lens.append(doc.shape[0])
    for doc in x_tst:
        doc_lens.append(doc.shape[0])

    doc_pad_len = round(sum(doc_lens) / len(doc_lens))
    doc_pad_len += 5

    # padding - now all docs have the same number of sentences
    # x_tr = pad_sequences(x_tr, doc_pad_len, padding='post',
    #                      truncating='post')
    # y_tr = pad_sequences(y_tr, doc_pad_len, padding='post',
    #                      truncating='post')
    x_tst = pad_sequences(x_tst, doc_pad_len, padding='post',
                          truncating='post')
    y_tst = pad_sequences(y_tst, doc_pad_len, padding='post',
                          truncating='post')

    # help determine pos_weight in rnn_model.custom_loss() correctly
    # percentage = get_percentage_of_segment_starting_sentences(y_tr, doc_pad_len)
    # print(f"Percentage of segment-starting sentences: {percentage}%")

    '''
    x_tr - all docs (shape=(num_docs_tr, num_sen_in_longest_doc,
                                        num_words_in_longest_sen))
    x_tr[i] - i-th doc
    x_tr[i][j] - j-th sentence in the i-th doc

    y_tr.shape = (num_docs_tr, num_sen_in_longest_doc)
    '''
    embedding_dim = 100
    embedding_matrix = calculate_embedding_matrix(
        tokenizer.word_index, embedding_dim)
    # num_docs_tr = x_tr.shape[0]
    num_docs_tst = x_tst.shape[0]
    num_sen_per_doc = len(x_tr[0])
    # num_sen_per_doc = 0
    # sen_len = x_tr.shape[2]
    sen_len = len(x_tr[0][0])

    batch_size = 32
    num_epochs = 30
    validation_split = 0.2
    # y_tr = y_tr.reshape(num_docs_tr, num_sen_per_doc, 1)
    # y_tst = y_tst.reshape(num_docs_tst, num_sen_per_doc, 1)
    y_tst = np.expand_dims(y_tst, axis=2)
    texts = [constants.TEXT_1, constants.TEXT_2,
             constants.TEXT_3, constants.TEXT_4, constants.TEXT_5]
    train = True
    model = None
    if train:
        model = train_model(
            x_tr, y_tr, batch_size, num_epochs, validation_split,
            save_model_path, tokenizer, embedding_matrix, embedding_dim,
            num_sen_per_doc, sen_len, num_docs)
        print(f"Evaluating model: [loss, accuracy]: {model.evaluate(x_tst, y_tst)}")
    else:
        ''' if loss is required (e.g. for model.evaluate()), use
        model = load_model(save_model_path,
        custom_objects={'custom_loss': rnn_model.custom_loss(y_true, y_pred)})
        if the model will only be used for inference, compile=False is your friend :)
        '''
        # compile=False added because of custom loss function
        model = load_model(save_model_path, compile=False)

    # seq_text = x_tr[0].reshape(1, x_tr[0].shape[0], x_tr[0].shape[1])
    # label = y_tr[0].reshape(1, y_tr[0].shape[0], y_tr[0].shape[1])
    # predictions = model.predict(seq_text)
    # print(predictions)
    # for i in range(len(predictions[0])):
    #     print(f"{predictions[0][i][0]}: {y_tr[0][i]}")
    # exit()

    # print("---------------------------------")
    # predictions1 = model.predict(x_tr[1])[0]
    # for i in range(len(predictions1)):
    #     print(f"{predictions1[i][0]}: {x_tr[1]:50}")

    evaluate_model(texts[4], tokenizer, sen_pad_len,
                   doc_pad_len, model)
