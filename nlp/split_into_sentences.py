import nltk
from constants import TEXT_1, TEXT_2, TEXT_3, TEXT_4
import re


def remove_digits(text, replace_with):
    new_text = ""
    for i in range(len(text)):
        if text[i].isdigit():
            new_text += replace_with
        else:
            new_text += text[i]
    return new_text


def regex(text):
    text = re.sub("http\S+", "URL", text)  # replace urls
    text = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}").sub(r"IPADDRESS", text)  # replace ip addresses
    text = remove_digits(text, "#")
    return text


def split_text_into_sentences(text):
    text = regex(text)
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
    print(len(sents))