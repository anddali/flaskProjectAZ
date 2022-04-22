from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

import re


import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def tokenize(text):
    return word_tokenize(text)


# remove links, words starting with @ or # and query text
def initial_cleaning(text, query):
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove words starting with @
    text = re.sub(r"@\S+", "", text)
    # remove words starting with #
    text = re.sub(r"#\S+", "", text)
    # remove query text
    for r in query:
        text = re.sub(fr"{r}", "", text, flags=re.IGNORECASE)
    return text


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    return ([token.lower() for token in text if token.lower() not in stop_words])


def remove_punct(text):
    punct = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    return ([char for char in text if char not in punct])


def remove_non_words(text):
    return ([w for w in text if w.isalpha()])


def remove_short_words(text, min_length):
    return ([w for w in text if len(w) >= min_length])


def stem_words(text):
    stemmer = WordNetLemmatizer()
    return ([stemmer.lemmatize(token) for token in text])


def stem_words_more(text):
    stemmer = PorterStemmer()
    return ([stemmer.stem(token) for token in text])


def process_tweet(text, query=[]):
    words = initial_cleaning(text, query)
    words = tokenize(words)
    words = remove_stop_words(words)
    words = remove_punct(words)
    words = remove_non_words(words)
    words = stem_words(words)
    words = remove_short_words(words, 3)  # min len of word 3
    return ' '.join(words)