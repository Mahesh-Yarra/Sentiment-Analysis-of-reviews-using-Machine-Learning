from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import nltk
import os
import pickle

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Specify the directory to save the models
MODELS_DIR = "./Models/"

# Check if the directory exists, if not, create it
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Paths for files
LEXICON_POSITIVE = r"./Datasets/positive-words.txt"
LEXICON_NEGATIVE = r"./Datasets/negative-words.txt"
LEXICON_CONNOTATION = r"./Datasets/connotations.csv"


def extract_features(df_subset):
    reviews = df_subset['review']
    # Initialize the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(reviews)
    tfidf_features_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Assuming X is your features DataFrame and y is your target DataFrame
    # Here X is tfidf_features_df and y is df_subset['sentiment']
    # Make sure X and y have the same number of rows

    # # Calculate correlation between each feature in X and the target variable in y
    # correlation = tfidf_features_df.corrwith(df_subset['sentiment'])

    # # Sort the correlation values and get the indices of the top correlated features
    # selected_features_8000 = correlation.abs().nlargest(8000).index

    # # Filter the features dataframe with selected features
    # selected_df_8000 = tfidf_features_df[selected_features_8000]

    # Appending connotation features
    connotations = pd.read_csv(LEXICON_CONNOTATION)
    word_emotion_map = dict(zip(connotations['word'], connotations['emotion']))

    pos_neg_conn_counts_df = pd.DataFrame([update_counts(review, word_emotion_map) for review in reviews],
                                          columns=['Positive_Connotation_Count', 'Negative_Connotation_Count'])

    # Appending positive and negative word's count
    # Load positive and negative words from files
    positive_words_df = pd.read_csv(LEXICON_POSITIVE, header=None, names=['words'])
    negative_words_df = pd.read_csv(LEXICON_NEGATIVE, header=None, names=['words'])

    # Convert DataFrame columns to sets
    positive_words = set(positive_words_df['words'].tolist())
    negative_words = set(negative_words_df['words'].tolist())

    pos_neg_counts_df = pd.DataFrame([update_word_counts(review, positive_words, negative_words) for review in reviews],
                                     columns=['Positive_Word_Count', 'Negative_Word_Count'])

    # Appending VADER features
    # Use VADER for sentiment analysis
    sid = SentimentIntensityAnalyzer()
    vader_scores_df = pd.DataFrame([vader_sentiment(review, sid) for review in reviews],
                                   columns=['Positive_VADER_Count', 'Negative_VADER_Count'])

    # Appending LDA topic models
    # Define the number of topics
    num_topics = 250

    # Create the LDA pipeline
    lda_pipeline = make_pipeline(
        CountVectorizer(),  # CountVectorizer converts text to a matrix of token counts
        TfidfTransformer(),  # TF-IDF transformation
        LatentDirichletAllocation(n_components=num_topics, random_state=42)  # LDA for topic modeling
    )

    # Fit and transform data using the LDA pipeline also creating DataFrame for LDA features
    x_lda = pd.DataFrame(lda_pipeline.fit_transform(reviews),
                         columns=[f"Topic_{i}" for i in range(1, num_topics + 1)])

    # Concatenate all features into a single DataFrame
    selected_features = pd.concat(
        [tfidf_features_df, pos_neg_conn_counts_df, pos_neg_counts_df, vader_scores_df, x_lda], axis=1)

    # Normalize features
    columns_to_normalize = ['Positive_Connotation_Count', 'Negative_Connotation_Count',
                            'Positive_Word_Count', 'Negative_Word_Count',
                            'Positive_VADER_Count', 'Negative_VADER_Count']
    selected_features_normalized = normalize_features(selected_features, columns_to_normalize)

    # Pickle TfidfVectorizer
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # Pickle the correlation_feature_selector
    # with open(os.path.join(MODELS_DIR, 'correlated_feature_picker.pkl'), 'wb') as f:
    #     pickle.dump(correlation, f)

    # Pickle LDA pipeline
    with open(os.path.join(MODELS_DIR, 'lda_topics_extractor.pkl'), 'wb') as f:
        pickle.dump(lda_pipeline, f)

    return selected_features_normalized


# Define helper functions for feature extraction
def update_counts(review, word_emotion_map):
    positive_count = sum(
        1 for word in review.split() if word in word_emotion_map and word_emotion_map[word] == 'positive')
    negative_count = sum(
        1 for word in review.split() if word in word_emotion_map and word_emotion_map[word] == 'negative')
    return positive_count, negative_count


def update_word_counts(review, positive_words, negative_words):
    positive_count = sum(1 for word in review.split() if word in positive_words)
    negative_count = sum(1 for word in review.split() if word in negative_words)
    return positive_count, negative_count


def vader_sentiment(review, sid):
    scores = sid.polarity_scores(review)
    return scores['pos'] * 100, scores['neg'] * 100


def normalize_features(features, columns_to_normalize):
    """
    Normalize specified columns in the features DataFrame using MinMaxScaler.

    Parameters:
    - features (DataFrame): DataFrame containing the features to be normalized.
    - columns_to_normalize (list): List of column names to be normalized.

    Returns:
    - normalized_features (DataFrame): DataFrame with specified columns normalized.
    """
    scaler = MinMaxScaler()
    normalized_features = features.copy()
    normalized_features[columns_to_normalize] = scaler.fit_transform(features[columns_to_normalize])
    return normalized_features


def scale_features(df):
    """
    Scale all features in the DataFrame using Min-Max scaling.

    Parameters:
    - df (pd.DataFrame): DataFrame containing features to be scaled.

    Returns:
    - pd.DataFrame: DataFrame with all features scaled.
    """
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df)
    return pd.DataFrame(scaled_features, columns=df.columns)
