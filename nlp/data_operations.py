import matplotlib.pyplot as plt
from text_preprocessing import parse_sentence

SEGMENT_DELIMITER = "========"


def plot_training_and_validation_data(train_acc, val_acc, train_loss,
                                      val_loss, num_rows):
    """
    plots training and validation curves
    :param train_acc:
    :param val_acc:
    :param train_loss:
    :param val_loss:
    :param num_rows:
    :return:
    """
    epochs = range(1, len(train_acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, train_acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.xlabel('Epochs')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.savefig('vanilla_cnn_'+num_rows+'_train_val_acc', bbox_inches='tight')

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('vanilla_cnn_'+num_rows+'_train_val_loss', bbox_inches='tight')

    plt.show()


def read_docs(path, num_docs=100):
    current_paragraph_x = []
    list_of_all_words = []
    sentence_document = {}  # sentence_index: document_index
    y_tr, y_tst = [], []
    starts_new_segment = False
    sentence_lengths = []
    document_index, sentence_index = -1, 0
    print(f"Loading {num_docs} documents...")
    with open(path, "rt", encoding="utf-8") as myfile:
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
            if line.startswith("\x00") or line.startswith("wiki_") \
                    or line.startswith("***LIST"):
                continue
            # start of subsection in section
            if len(current_paragraph_x) == 0 \
                    and line.startswith(SEGMENT_DELIMITER):
                continue
            line = parse_sentence(line)
            if document_index > 0.8 * num_docs:
                # denotes new segment title
                if line.startswith(SEGMENT_DELIMITER):
                    current_paragraph_x = []
                    # the next line starts a new segment
                    starts_new_segment = True
                    continue    # no need to add segment title to data
                if starts_new_segment:
                    current_paragraph_x.append(line)
                    y_tst.append(1)
                    starts_new_segment = False
                else:
                    y_tst.append(0)
                    current_paragraph_x.append(line)
            else:
                # denotes new segment title
                if line.startswith(SEGMENT_DELIMITER):
                    current_paragraph_x = []
                    # the next line starts a new segment
                    starts_new_segment = True
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


if __name__ == '__main__':
    y_tr, y_tst, list_of_all_words, sentence_document, \
        avg_sen_len = read_docs("extracted/wiki_727K", 1000)
    # print(len(list_of_all_words))
    # print(list_of_all_words)
