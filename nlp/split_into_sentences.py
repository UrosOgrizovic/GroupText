import nltk
from constants import TEXT_1, TEXT_2, TEXT_3, TEXT_4
import re
from collections import Counter
from string import printable


UPPERCASE_LETTERS = [chr(i) for i in range(65, 91)]
LOWERCASE_LETTERS = [chr(i) for i in range(97, 123)]
LETTERS = UPPERCASE_LETTERS + LOWERCASE_LETTERS
LEMMATIZATION_EXCEPTIONS = ["was", "as"]
STOP_WORDS = Counter(nltk.corpus.stopwords.words('english'))


def replace_digits(text, replace_with="#"):
    """Replace digits with a specific character.

    Args:
        text (string): text to modify
        replace_with (str, optional): character to replace with. Defaults to "#".

    Returns:
        new_text: modified text
    """
    new_text = ""
    for i in range(len(text)):
        if text[i].isdigit():
            new_text += replace_with
        else:
            new_text += text[i]
    return new_text


def regex(text):
    """Replace urls, ip addresses and digits

    Args:
        text (string): text to modify

    Returns:
        text: modified text
    """
    text = re.sub(r"http\S+", "URL", text)  # replace urls
    text = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}").sub(r"IPADDRESS", text)  # replace ip addresses
    text = replace_digits(text, "#")
    return text


def replace_newline(text):
    """Replace "\n" and "-\n" in text with a space.

    Args:
        text (string): text to modify

    Returns:
        string: modified text
    """
    text = text.replace("-\n", " ")
    text = text.replace("\n", " ")
    return text


def split_text_into_sentences(text):
    sents = nltk.tokenize.sent_tokenize(text)
    for i in range(len(sents)):
        sents[i] = replace_newline(sents[i])
    for i in range(len(sents)):
        sents[i] = lemmatization(sents[i])
    return sents


def parse_sentence(sent):
    """Regex, replace newline and lemmatization on a single sentence.

    Args:
        sent (string): sentence to modify

    Returns:
        sent: modified sent
    """
    sent = regex(sent)
    sent = replace_newline(sent)
    return lemmatization(sent)


def lemmatization(sent):
    """Lemmatize sentence, removew stop words and non-english words

    Args:
        sent (sent): sentence to modify

    Returns:
        string: modified sentence
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    new_sent = ""
    sent = sent.split(" ")
    for i in range(len(sent)):
        word = sent[i]
        filtered_word =  ''.join(filter(lambda x: x in printable, word))
        # eliminate stop words and non-english words, such as ὀρθός
        if len(word) > 0 and word not in STOP_WORDS and len(filtered_word) == len(word):
            punctuation = ""
            
            if word[-1] not in LETTERS: # if the word ends in a punctuation mark
                punctuation = word[-1]
                word = word[:-1]
            word = lemmatizer.lemmatize(word)
            new_sent += word + punctuation + " "
            
    return new_sent


if __name__ == "__main__":
    text = regex(TEXT_3)
    text = text.lower()
    sents = split_text_into_sentences(text)
    for sent in sents:
        print("s:", sent)
    print("--")
    print(len(sents))