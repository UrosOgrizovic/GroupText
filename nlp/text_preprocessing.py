import nltk
from constants import TEXT_3
import re
from collections import Counter
from string import printable
from unidecode import unidecode

UPPERCASE_LETTERS = [chr(i) for i in range(65, 91)]
LOWERCASE_LETTERS = [chr(i) for i in range(97, 123)]
LETTERS = UPPERCASE_LETTERS + LOWERCASE_LETTERS
LEMMATIZATION_EXCEPTIONS = ["was", "as"]
STOP_WORDS = Counter(nltk.corpus.stopwords.words('english'))


def replace_digits(text, replace_with="#"):
    """Replace digits with a specific character.

    Args:
        text (string): text to modify
        replace_with (str, optional): character to replace with. Defaults to
        "#".

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
    text = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}").sub(
        r"IPADDRESS", text)  # replace ip addresses
    # text = replace_digits(text, "#")
    text = ''.join(char for char in text if not char.isdigit()) # remove digits
    text = text.replace(' s ', '')  # left over from e.g. "90s"
    text = text.replace(' \'s ', '')    # left over from e.g. "'90s"
    text = text.replace(' % ', '')  # left over from e.g. "50%"
    text = text.replace(' . ', '')  # left over from e.g. "45.32"
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


def remove_accented_chars(text):
    """Remove accented characters from text, e.g. café

    Args:
        text (string): text to process

    Returns:
        string: processed text
    """
    text = unidecode(text)
    return text


def split_text_into_sentences(text):
    return nltk.tokenize.sent_tokenize(text)


def parse_sentence(sent):
    """Remove accented characters, regex,
        replace newline and lemmatization
        on a single sentence.

    Args:
        sent (string): sentence to modify

    Returns:
        sent: modified sent
    """
    sent = remove_accented_chars(sent)
    sent = regex(sent)
    sent = replace_newline(sent)
    return sent
    # return lemmatization(sent)


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
        filtered_word = ''.join(filter(lambda x: x in printable, word))
        # eliminate stop words and non-english words, such as ὀρθός
        if len(word) > 0 and word not in STOP_WORDS and len(filtered_word) == \
                len(word):
            punctuation = ""

            # if the word ends in a punctuation mark
            if word[-1] not in LETTERS:
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
