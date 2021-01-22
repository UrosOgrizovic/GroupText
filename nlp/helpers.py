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


def find_lengths_in_batch(x):
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
        for sen in doc:
            avg_sen_len += len(sen)
            num_sents += 1
    avg_doc_len /= len(x)
    avg_sen_len /= num_sents
    return round(avg_doc_len), round(avg_sen_len)
    # max_doc_len = 0
    # max_sen_len = 0
    # for doc in x:
    #     if len(doc) > max_doc_len:
    #         max_doc_len = len(doc)
    #     for sen in doc:
    #         if len(sen) > max_sen_len:
    #             max_sen_len = len(sen)
    # return max_doc_len, max_sen_len
