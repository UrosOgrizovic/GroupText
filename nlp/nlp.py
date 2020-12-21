import rnn_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from constants import TEXT_1, TEXT_2, TEXT_3, TEXT_4
import numpy as np
import os
from split_into_sentences import split_text_into_sentences
from tensorflow.keras.models import load_model


def calculate_embedding_matrix(word_index, embedding_dim):
    embeddings_index = {}
    f = open(os.path.join("glove", 'glove.6B.100d.txt'), encoding="utf-8")
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


if __name__ == "__main__":
    num_rows, num_words = 10000, 1000
    lines = []
    tokenizer = Tokenizer(num_words=num_words)
    x_tr, y_tr, x_tst, y_tst = [], [], [], []
    max_len_line = 0
    starts_new_segment = False
    with open("extracted/wiki_727K", "rt", encoding="utf-8") as myfile:
        for i in range(num_rows):
            line = next(myfile)[:-2]
            # skip these lines, I'm not sure if they are meaningful or metadata
            if line.startswith("\x00") or line.startswith("wiki_") or line.startswith("***LIST"):
                continue
            max_len_line = len(line) if len(line) > max_len_line else max_len_line
            if i > 0.8 * num_rows:
                if line.startswith("========"):     # denotes new segment title
                    starts_new_segment = True   # the next line starts a new segment
                    continue    # no need to add segment title to data
                if starts_new_segment:
                    y_tst.append(1)     # this line starts a segment
                    starts_new_segment = False
                else:
                    y_tst.append(0)
                x_tst.append(line)
            else:
                if line.startswith("========"):     # denotes new segment title
                    starts_new_segment = True   # the next line starts a new segment
                    continue    # no need to add segment title to data
                if starts_new_segment:
                    y_tr.append(1)      # this line starts a segment
                    starts_new_segment = False
                else:
                    y_tr.append(0)
                x_tr.append(line)
            lines.append(line)

    # convert strings to numbers
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    # perform padding
    sequences = pad_sequences(sequences, max_len_line)

    # 80/20 train-test split
    x_tr, x_tst = sequences[:len(y_tr)], sequences[len(y_tr):]
    # one-hot encode labels
    y_tr, y_tst = to_categorical(y_tr), to_categorical(y_tst)
    # print(len(x_tr), len(y_tr), len(x_tst), len(y_tst))
    # print(x_tr, y_tr)
    # print('--')
    # print(x_tst, y_tst)

    embedding_matrix_dim = 100
    embedding_matrix = calculate_embedding_matrix(tokenizer.word_index, embedding_matrix_dim)
    save_model_path = "saved_models/model" + str(num_rows) + ".h5"

    model = rnn_model.create_model(len(tokenizer.word_index), embedding_matrix, embedding_matrix_dim)
    model.fit(x_tr, y_tr, epochs=3, batch_size=32)
    model.evaluate(x_tst, y_tst)
    model.save(save_model_path)

    # model = load_model(save_model_path)
    texts = [TEXT_1, TEXT_2, TEXT_3, TEXT_4]
    text = texts[0]
    text = split_text_into_sentences(text)
    # for i in range(len(text)):
    #     print("#" + str(i) + ":", text[i])
    seq_text = tokenizer.texts_to_sequences(text)
    seq_text = pad_sequences(seq_text, max_len_line)
    predictions = model.predict(seq_text)
    print(predictions)
    print("The model thinks each of the following sentences starts a new section:")
    for i in range(len(predictions)):
        if predictions[i][0] < predictions[i][1]:   # e.g. [0 1], i.e. new segment
            print("-", text[i])
