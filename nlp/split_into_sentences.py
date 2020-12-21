import nltk
from constants import TEXT_1, TEXT_2, TEXT_3, TEXT_4


def split_text_into_sentences(text):
    sents = nltk.tokenize.sent_tokenize(text)
    for i in range(len(sents)):
        sents[i] = sents[i].replace("-\n", "")
        sents[i] = sents[i].replace("\n", " ")
    return sents


if __name__ == "__main__":
    sents = split_text_into_sentences(TEXT_2)
    for sent in sents:
        print("s:", sent)
    print("--")
    print(len(sentences))