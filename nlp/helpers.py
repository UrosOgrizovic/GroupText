import numpy as np
import os


def abbreviate_num_to_str(num):
    if num >= 100000:
        num = int(round(num / 100000, 2) * 100)
    elif num >= 10000:
        num = int(round(num / 10000, 1) * 10)
    elif num >= 1000:
        num = int(round(num / 1000))
    else:
        return str(num)
    return f'{num}k'


def find_avg_doc_length_in_batch(x):
    """Find average document and sentence
    length in batch.

    Args:
        x (list): batch of documents

    Returns:
        tuple: avg_doc_len, avg_sen_len
    """

    avg_doc_len = 0
    avg_sen_len = 0
    num_sents = 0
    for doc in x:
        avg_doc_len += len(doc)
    avg_doc_len /= len(x)
    if avg_doc_len > 80:
        # capping doc len to prevent` OOM
        avg_doc_len = 80
    return round(avg_doc_len)


def get_percentage_of_segment_starting_sentences(y, doc_pad_len):
    num_ones = 0
    for doc in y:
        for i in range(doc_pad_len):
            if doc[i] == 1:
                num_ones += 1
    total_num_sen_y = doc_pad_len * len(y)
    ratio = total_num_sen_y / num_ones
    percentage = round(1 / ratio, 2) * 100
    # print(f"Total number of sentences in y {total_num_sen_y}")
    # print(f"Number of segment-starting sentences in y {num_ones}")
    # print(f"Non-segment starting sentences/segment-starting sentences {ratio}")
    return percentage


def get_path(num_docs, set='tr'):
    """Gets path to preprocessed data.

    Args:
        num_docs (int)
        set (str, optional): 'tr', 'val' or 'tst'. Defaults to 'tr'.

    Returns:
        str
    """
    num = abbreviate_num_to_str(num_docs)
    return f'data/dump_{set}_{num}.pkl'
