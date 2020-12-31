import rnn_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import constants
import numpy as np
import os
import gc
import random
from split_into_sentences import split_text_into_sentences, regex
from tensorflow.keras.models import load_model
import data_operations
import re

SEGMENT_DELIMITER = "========"


def calculate_embedding_matrix(word_index, embedding_dim):
    embeddings_index = {}
    f = open(os.path.join("glove", 'glove.6B.' + str(embedding_dim) + 'd.txt'), encoding="utf-8")
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


def train_model(x_tr, y_tr, batch_size, num_epochs, validation_split, save_model_path, tokenizer, embedding_matrix,
                embedding_dim, num_sen_per_doc, sen_len):
    print("Creating model...")
    model = rnn_model.create_model(len(tokenizer.word_index), embedding_matrix, embedding_dim, num_sen_per_doc, sen_len)
    print("Finished creating model")
    # print(model.summary())
    # history = rnn_model.train_model(model, save_model_path, x_tr, y_tr, batch_size, num_epochs, validation_split)
    model.fit(x_tr, y_tr, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)
    model.save(save_model_path)
    # print(model.evaluate(x_tst, y_tst))
    # train_acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # data_operations.plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss, num_rows)
    print(f"Finished training and model saved to path {save_model_path}")


def evaluate_model(x_tst, y_tst, save_model_path, text, tokenizer, max_len):
    model = load_model(save_model_path)
    # print(f"Evaluating model: [loss, accuracy]: {model.evaluate(x_tst, y_tst)}")

    text = split_text_into_sentences(text)
    # for i in range(len(text)):
    #     print(f"#{str(i)}: {text[i]}")
    seq_text = tokenizer.texts_to_sequences(text)
    seq_text = pad_sequences(seq_text, max_len)
    seq_text = seq_text.reshape(1, seq_text.shape[0], seq_text.shape[1])
    len_before_padding = seq_text.shape[1]
    seq_text = pad_sequences(seq_text, longest_doc_len)

    predictions = model.predict(seq_text)[0]
    predictions = predictions[-len_before_padding:]  # not interested in the predictions for the padded elements
    print(predictions)
    # print("The model thinks each of the following sentences starts a new section:")
    for i in range(len(predictions)):
        print(f"{predictions[i][0]}: {text[i]:50}")
        # if predictions[i][0] > 0.5:
        #     print(f"-{text[i]}")


def read_docs(num_docs=100):
    current_paragraph_x = []
    list_of_all_words = []
    sentence_document = {}  # sentence_index: document_index
    y_tr, y_tst = [], []
    starts_new_segment = False
    sentence_lengths = []
    document_index, sentence_index = -1, 0
    print(f"Loading {num_docs} documents...")
    with open("extracted/wiki_727K", "rt", encoding="utf-8") as myfile:
        while document_index < num_docs:
            line = next(myfile)[:-2]
            # new document, increase document_index
            if "1,preface" in line:
                document_index += 1
                starts_new_segment = True  # the next line starts a new segment
                if current_paragraph_x:
                    current_paragraph_x = []
                continue
            # skip these lines, I'm not sure if they are meaningful or metadata
            if line.startswith("\x00") or line.startswith("wiki_") or line.startswith("***LIST"):
                continue
            # start of subsection in section
            if len(current_paragraph_x) == 0 and line.startswith(SEGMENT_DELIMITER):
                continue
            line = regex(line)
            if document_index > 0.8 * num_docs:
                if line.startswith(SEGMENT_DELIMITER):     # denotes new segment title
                    current_paragraph_x = []
                    starts_new_segment = True   # the next line starts a new segment
                    continue    # no need to add segment title to data
                if starts_new_segment:
                    current_paragraph_x.append(line)
                    y_tst.append(1)
                    starts_new_segment = False
                else:
                    y_tst.append(0)
                    current_paragraph_x.append(line)
            else:
                if line.startswith(SEGMENT_DELIMITER):     # denotes new segment title
                    current_paragraph_x = []
                    starts_new_segment = True   # the next line starts a new segment
                    continue    # no need to add segment title to data
                if starts_new_segment:
                    y_tr.append(1)
                    current_paragraph_x.append(line)
                    starts_new_segment = False
                else:
                    y_tr.append(0)
                    current_paragraph_x.append(line)
            list_of_all_words.append(line.split(" "))
            sentence_document[sentence_index] = document_index
            sentence_index += 1
            sentence_lengths.append(len(line.split(" ")))
    print("Finished loading documents")
    avg_sen_len = round(sum(sentence_lengths) / len(sentence_lengths))
    return y_tr, y_tst, list_of_all_words, sentence_document, avg_sen_len


if __name__ == "__main__":
    num_docs, num_rows = 100, 10000
    save_model_path = "saved_models/model" + str(num_docs) + ".h5"
    y_tr, y_tst, list_of_all_words, sentence_document_mapping, avg_sen_len = read_docs(num_docs)
    # num_words = 5000
    # convert strings to numbers
    tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(collection)
    print("Fitting tokenizer...")
    tokenizer.fit_on_texts(list_of_all_words)
    print("Finished fitting tokenizer")

    sequences = tokenizer.texts_to_sequences(list_of_all_words)
    del list_of_all_words
    gc.collect()

    max_len = avg_sen_len + 5

    # perform padding/truncating (truncate ends of sentences, because beginnings are more important for this task)
    sequences = pad_sequences(sequences, max_len, truncating='post')
    # 80/20 train-test split
    x_tr, x_tst = sequences[:len(y_tr)], sequences[len(y_tr):]

    del sequences
    gc.collect()

    # group data as documents
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

    curr_doc_x, curr_doc_y = [x_tst[0]], [y_tst[0]]
    for i in range(1, len(x_tst)):
        # new document
        if sentence_document_mapping[i + len(x_tr)-1] != sentence_document_mapping[i-1+len(x_tr)-1]:
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

    documents_x_tr = np.array([np.array(lst, dtype='float') for lst in documents_x_tr], dtype='object')
    documents_x_tst = np.array([np.array(lst, dtype='float') for lst in documents_x_tst], dtype='object')

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

    longest_doc_len = 0
    for doc in x_tr:
        if doc.shape[0] > longest_doc_len:
            longest_doc_len = doc.shape[0]
    for doc in x_tst:
        if doc.shape[0] > longest_doc_len:
            longest_doc_len = doc.shape[0]

    x_tr = pad_sequences(x_tr, longest_doc_len)     # now all docs have the same number of sentences
    y_tr = pad_sequences(y_tr, longest_doc_len)
    x_tst = pad_sequences(x_tst, longest_doc_len)
    y_tst = pad_sequences(y_tst, longest_doc_len)
    '''
    x_tr - all docs (shape=(num_docs_tr, num_sen_in_longest_doc, num_words_in_longest_sen))
    x_tr[i] - i-th doc
    x_tr[i][j] - j-th sentence in the i-th doc
    
    y_tr.shape = (num_docs_tr, num_sen_in_longest_doc)
    '''

    embedding_dim = 50
    embedding_matrix = calculate_embedding_matrix(tokenizer.word_index, embedding_dim)

    num_docs_tr = x_tr.shape[0]
    num_docs_tst = x_tst.shape[0]
    num_sen_per_doc = x_tr.shape[1]
    sen_len = x_tr.shape[2]

    batch_size = 32
    num_epochs = 3
    validation_split = 0.1
    y_tr = y_tr.reshape(num_docs_tr, num_sen_per_doc, 1)
    y_tst = y_tst.reshape(num_docs_tst, num_sen_per_doc, 1)

    texts = [constants.TEXT_1, constants.TEXT_2, constants.TEXT_3, constants.TEXT_4, constants.TEXT_5]
    train = False
    if train:
        train_model(x_tr, y_tr, batch_size, num_epochs, validation_split, save_model_path, tokenizer, embedding_matrix,
                embedding_dim, num_sen_per_doc, sen_len)
    else:
        evaluate_model(x_tst, y_tst, save_model_path, texts[4], tokenizer, max_len)
