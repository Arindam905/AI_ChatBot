import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenizer(word):
    return nltk.word_tokenize(word)


def stem(word):
    return stemmer.stem(word)


def bag_of_words(word):
    pass


