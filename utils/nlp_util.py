import nltk
from nltk.corpus import stopwords, wordnet

from nltk import word_tokenize
from nltk.corpus import wordnet as wn
import string


def parse_entity(text):
    stops = set(stopwords.words("english"))
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word not in stops]
    words = [wordnet.morphy(word) for word in words if word not in stops]
    return words


def get_synonyms(words):
    res = set()
    for word in words:
        syn_set = wn.synsets(word)
        try:
            syn_answer_list = syn_set[0].lemma_names()
        except:
            syn_answer_list = [word]
        res.update(syn_answer_list)
    return res


def tokenize_list_of_names(syn_answer_list):
    res = []
    for word in syn_answer_list:
        res.extend(word.split(" "))
        res.extend(word.split("-"))
        res.extend(word.split("_"))

    res = list(set(res))
    # lower
    res = [word.lower() for word in res]
    return res


def class2list(class_name):
    class_names = class_name.strip().split(",")
    class_names = [class_name.strip() for class_name in class_names]
    return class_names
