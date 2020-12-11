from nltk.corpus import stopwords
import re
from contraction_mapping import contraction_mapping

stop_words = stopwords.words('english')


def story_cleaner(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'\([^)]*\)', '', sentence)
    sentence = re.sub(r'"', '', sentence)
    sentence = ''.join([contraction_mapping[t] if t in contraction_mapping else t for t in sentence])
    sentence = re.sub(r"'s\b", "", sentence)
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    tokens = [w for w in sentence.split() if not w in stop_words]
    long_words = [w for w in tokens if len(w) > 3]
    sentence = ''
    for items in long_words:
        sentence += items + ' '
    return sentence


def summary_cleaner(sentence):
    sentence = '<BREAK>'.join([item for item in sentence])
    sentence = sentence.lower()
    sentence = re.sub(r'"', '', sentence)
    sentence = ''.join([contraction_mapping[t] if t in contraction_mapping else t for t in sentence])
    sentence = re.sub(r"'s\b", "", sentence)
    sentence = '_START_'+ re.sub(r'[^a-zA-Z]<>', ' ', sentence)+ ' _END_'
    tokens = sentence.split()
    sentence = ''
    for items in tokens:
        sentence += items+' '
    return sentence

