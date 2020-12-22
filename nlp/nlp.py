import rnn_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from constants import TEXT_1, TEXT_2, TEXT_3, TEXT_4
import numpy as np
import os
import gc
import random
from split_into_sentences import split_text_into_sentences
from tensorflow.keras.models import load_model
import data_operations

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


if __name__ == "__main__":
    num_rows, num_words = 50000, 1000
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
                    if random.random() > 0.9:   # attempting to create a more balanced dataset
                        y_tst.append(0)
                    else:
                        continue
                x_tst.append(line)
            else:
                if line.startswith("========"):     # denotes new segment title
                    starts_new_segment = True   # the next line starts a new segment
                    continue    # no need to add segment title to data
                if starts_new_segment:
                    y_tr.append(1)      # this line starts a segment
                    starts_new_segment = False
                else:
                    if random.random() > 0.9:   # attempting to create a more balanced dataset
                        y_tr.append(0)
                    else:
                        continue
                x_tr.append(line)
            lines.append(line)

    # convert strings to numbers
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    del lines
    gc.collect()

    # most sentences aren't that long
    max_len_line = max_len_line // 2
    # perform padding
    sequences = pad_sequences(sequences, max_len_line)

    # 80/20 train-test split
    x_tr, x_tst = sequences[:len(y_tr)], sequences[len(y_tr):]
    # one-hot encode labels - no need to do it if Dense(1) is the final layer
    # y_tr, y_tst = to_categorical(y_tr), to_categorical(y_tst)

    del sequences
    gc.collect()

    # print(len(x_tr), len(y_tr), len(x_tst), len(y_tst))
    # print(x_tr, y_tr)
    # print('--')
    # print(x_tst, y_tst)

    embedding_dim = 300
    embedding_matrix = calculate_embedding_matrix(tokenizer.word_index, embedding_dim)
    save_model_path = "saved_models/model" + str(num_rows) + ".h5"

    model = rnn_model.create_model(len(tokenizer.word_index), embedding_matrix, embedding_dim)
    batch_size = 32
    num_epochs = 20
    validation_split = 0.1
    history = rnn_model.train_model(model, save_model_path, x_tr, y_tr, batch_size, num_epochs, validation_split)
    # model.fit(x_tr, y_tr, epochs=10, batch_size=64)
    print(model.evaluate(x_tst, y_tst))
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    data_operations.plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss, num_rows)
    # model.save(save_model_path)

    # model = load_model(save_model_path)
    texts = [TEXT_1, TEXT_2, TEXT_3, TEXT_4]
    text = texts[1]
    text = split_text_into_sentences(text)
    for i in range(len(text)):
        print("#" + str(i) + ":", text[i])
    seq_text = tokenizer.texts_to_sequences(text)
    seq_text = pad_sequences(seq_text, max_len_line)
    predictions = model.predict(seq_text)
    print(predictions)
    print("The model thinks each of the following sentences starts a new section:")
    for i in range(len(predictions)):
        # if predictions[i][0] < predictions[i][1]:   # e.g. [0 1], i.e. new segment
        if predictions[i] > 0.5:   # e.g. [0 1], i.e. new segment
            print("-", text[i])
