import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Suppress the specific warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Downloading required resources for NLTK
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Define stopwords
stop_words = set(stopwords.words('english'))


# Functions for text preprocessing
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.sub(url_pattern, '', text)


def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


def remove_extra_whitespaces(text):
    return re.sub(r'\s+', ' ', text).strip()


# Function to get the part of speech for WordNet lemmatizer
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
        return wordnet.NOUN  # Default to noun if the part of speech is not found


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return ' '.join(lemmatized_tokens)


# Comprehensive preprocess function including punctuation removal
def preprocess(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuations using the specific provided code
    text = text.replace('[{}]'.format(string.punctuation), '')

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove URLs
    text = remove_urls(text)

    # Remove HTML tags
    text = remove_html_tags(text)

    # Remove stopwords
    text = remove_stopwords(text)

    # Clean non-alphanumeric characters
    text = clean_text(text)

    # Remove extra whitespaces
    text = remove_extra_whitespaces(text)

    # Lemmatize the text
    text = lemmatize_text(text)

    return text