import matplotlib.pyplot as plt
from text_preprocessing import parse_sentence
import pickle
import helpers
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences

SEGMENT_DELIMITER = "========"


def plot_training_and_validation_data(history, num_docs):
    """
    plots training and validation curves
    :param history:
    :param num_docs:
    :return:
    """
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, train_acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.xlabel('Epochs')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.savefig('learning_curves/model_'+num_docs+'_train_val_acc.png', bbox_inches='tight')

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('learning_curves/model_'+num_docs+'_train_val_loss.png', bbox_inches='tight')


def read_docs_in_batches(path, batch_size, curr_pos, num_in_path, sentence_document_mapping,
                        list_of_all_words, test_split, y_tr, y_tst, sentence_index, document_index,
                        num_docs_to_read):
    current_paragraph_x = []
    starts_new_segment = False
    batch_index = 0
    print(f"Loading {batch_size} documents...")
    print(f'Curr pos {curr_pos}')
    with open(path, "rt", encoding="utf-8") as myfile:
        myfile.seek(curr_pos, 0)     # move to where reading was stopped last time
        while batch_index < batch_size:
            # line = next(myfile)[:-2]
            line = myfile.readline()[:-2]
            # new document, increase document_index
            if "1,preface" in line:
                document_index += 1
                batch_index += 1
                starts_new_segment = True  # the next line starts a new segment
                if current_paragraph_x:
                    current_paragraph_x = []
                continue
            # skip these lines, I'm not sure if they are meaningful or metadata
            elif line.startswith("\x00") or line.startswith("wiki_") \
                    or line.startswith("***LIST"):
                continue
            # start of subsection in section
            elif len(current_paragraph_x) == 0 \
                    and line.startswith(SEGMENT_DELIMITER):
                continue
            line = parse_sentence(line)
            # denotes new segment title
            if line.startswith(SEGMENT_DELIMITER):
                current_paragraph_x = []
                # the next line starts a new segment
                starts_new_segment = True
                continue    # no need to add segment title to data
            elif starts_new_segment:
                current_paragraph_x.append(line)
                starts_new_segment = False
                if document_index > test_split * num_docs_to_read:
                    y_tst.append(1)
                else:
                    y_tr.append(1)
            else:
                current_paragraph_x.append(line)
                if document_index > test_split * num_docs_to_read:
                    y_tst.append(0)
                else:
                    y_tr.append(0)
            list_of_all_words.append(line.split(" "))
            sentence_document_mapping[sentence_index] = document_index
            sentence_index += 1
        curr_pos = myfile.tell()    # get position after reading
    print("Finished loading documents")
    # dump_to_path(list_of_all_words, f'list_of_all_words_{num_in_path}.pkl', 'ab')
    return y_tr, y_tst, list_of_all_words, sentence_document_mapping, curr_pos, sentence_index, document_index


def read_dumps_in_batches(path, batch_size, curr_pos):
    x, y = [], []
    batch_idx = 0
    with open(path, 'rb') as f:
        f.seek(curr_pos, 0)     # move to where reading was stopped last time
        unpickler = pickle.Unpickler(f)
        while batch_idx < batch_size:
            x.append(unpickler.load())
            y.append(unpickler.load())
            batch_idx += 1
        curr_pos = f.tell()     # get position after reading
    return x, y, curr_pos


def dump_to_path(object, path, mode='wb'):
    """Dump object to file via pickle

    Args:
        object ([type]): [description]
        path ([type]): [description]
        mode (str, optional): write or append ('wb' or 'ab'). Defaults to 'wb'.
    """
    with open(path, mode) as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)


def load_from_path(path):
    try:
        with open(path, 'rb') as input:
            object = pickle.load(input)
    except:
        object = None
    return object


def dump_x_y(x, y, path):
    with open(path, 'wb') as f:
        for x_el, y_el in zip(x, y):
            # incrementally dumping so as to allow incremental loading
            pickle.dump(x_el, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(y_el, f, pickle.HIGHEST_PROTOCOL)


def load_x_y(path):
    x, y = [], []
    with open(path, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        while True:
            try:
                x.append(np.array(unpickler.load()))
                y.append(np.array(unpickler.load()))
            except: # EOF
                break
    return np.array(x), np.array(y)


def get_tokenizer(path, list_of_all_words, num_words_to_keep):
    tokenizer = load_from_path(path)
    if tokenizer is None:
        # convert strings to numbers
        # tokenizer = Tokenizer(num_words=num_words_to_keep)
        tokenizer = Tokenizer()
        print("Fitting tokenizer...")
        tokenizer.fit_on_texts(list_of_all_words)
        print("Finished fitting tokenizer")
        print(f'Tokenizer word index len: {len(tokenizer.word_index)}')
        dump_to_path(tokenizer, path)
    return tokenizer


def process_loaded_docs(y_tr, y_tst, list_of_all_words, sentence_document_mapping, sen_pad_len,
                        tokenizer, num_in_path):
    """Processes loaded data and dumps it via pickle.

    Args:
        y_tr (list): [description]
        y_tst (list): [description]
        list_of_all_words (list): [description]
        sentence_document_mapping (dict): [description]
        sen_pad_len (int): [description]
        tokenizer (keras.preprocessing.text.Tokenizer)
        num_in_path (str): e.g. '1k', '10k', '100k'
    """
    sequences = tokenizer.texts_to_sequences(list_of_all_words)
    del list_of_all_words

    # perform sentence-level padding/truncating (truncate ends of sentences,
    # because beginnings are more important for this task)
    sequences = pad_sequences(sequences, sen_pad_len, padding='post',
                              truncating='post')
    # 80/20 train-test split
    x_tr, x_tst = sequences[:len(y_tr)], sequences[len(y_tr):]

    del sequences

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

    cutoff_point = round(0.8*len(x_tr))
    x_val = x_tr[cutoff_point:]
    y_val = y_tr[cutoff_point:]
    x_tr = x_tr[:cutoff_point]
    y_tr = y_tr[:cutoff_point]

    dump_x_y(x_tr, y_tr, f'data/dump_tr_{num_in_path}.pkl')
    dump_x_y(x_val, y_val, f'data/dump_val_{num_in_path}.pkl')
    dump_x_y(x_tst, y_tst, f'data/dump_tst_{num_in_path}.pkl')

    # help determine pos_weight in rnn_model.custom_loss() correctly
    # percentage = helpers.get_percentage_of_segment_starting_sentences(y_tr, doc_pad_len)
    # print(f"Percentage of segment-starting sentences: {percentage}%")

    '''
    x_tr.shape= (num_docs_tr, doc_len, sen_pad_len))
    x_tr[i] - i-th doc
    x_tr[i][j] - j-th sentence in the i-th doc

    y_tr.shape = (num_docs_tr, doc_len)

    doc_len gets padded for each batch in custom_generator() in rnn_model.py
    '''
    return x_tr, y_tr, x_val, y_val, x_tst, y_tst


if __name__ == '__main__':
    num_docs = 1000
    num_in_path = helpers.abbreviate_num_to_str(num_docs)

    x, y = load_x_y(f'data/dump_tr_{num_in_path}.pkl')
    print(x[1])
    print(y[1])
    print(len(x), len(y))
