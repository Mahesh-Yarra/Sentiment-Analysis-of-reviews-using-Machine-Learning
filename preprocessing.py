import re
import string

import nltk
from bs4 import BeautifulSoup
from langdetect import detect
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def preprocess(text):

    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Lowercase the text
    text = text.lower()

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word.lower() not in stop_words])

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    text = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags])

    return text
