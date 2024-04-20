import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model

from feature_engineering import update_counts, update_word_counts, vader_sentiment, normalize_features, scale_features
from preprocessing import preprocess

app = Flask(__name__)
CORS(app)

# Paths

# Paths for files
LEXICON_POSITIVE = r"./Datasets/positive-words.txt"
LEXICON_NEGATIVE = r"./Datasets/negative-words.txt"
LEXICON_CONNOTATION = r"./Datasets/connotations.csv"

# Load the trained model and necessary preprocessing steps
with open('./Models/tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('./Models/lda_topics_extractor.pkl', 'rb') as file:
    lda_pipeline = pickle.load(file)

# Load the scaler
with open('./Models/scaler.pkl', "rb") as file:
    scaler = pickle.load(file)

# Load the saved model
model = load_model('./Models/sentiment_model.h5')
num_topics = 250
connotations = pd.read_csv(LEXICON_CONNOTATION)
word_emotion_map = dict(zip(connotations['word'], connotations['emotion']))

positive_words_df = pd.read_csv(LEXICON_POSITIVE, header=None, names=['words'])
negative_words_df = pd.read_csv(LEXICON_NEGATIVE, header=None, names=['words'])
positive_words = set(positive_words_df['words'].tolist())
negative_words = set(negative_words_df['words'].tolist())


# Define route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from request
    data = request.json
    input_text = data.get('text', '')

    # Preprocess input text
    preprocessed_text = preprocess(input_text)

    # Vectorize preprocessed text
    vectorized_text = vectorizer.transform([preprocessed_text])
    vectorized_text_df = pd.DataFrame(vectorized_text.toarray(), columns=vectorizer.get_feature_names_out())

    # Transform text using LDA pipeline
    lda_features = lda_pipeline.transform([preprocessed_text])

    # Convert the array into a DataFrame
    lda_features_df = pd.DataFrame(lda_features,  columns=[f"Topic_{i}" for i in range(1, num_topics + 1)])

    # Extract additional features
    pos_neg_conn_counts_df = pd.DataFrame([update_counts(preprocessed_text, word_emotion_map)],
                                          columns=['Positive_Connotation_Count', 'Negative_Connotation_Count'])

    # Appending positive and negative word's count
    pos_neg_counts_df = pd.DataFrame([update_word_counts(preprocessed_text, positive_words, negative_words)],
                                     columns=['Positive_Word_Count', 'Negative_Word_Count'])

    # Appending VADER features
    # Use VADER for sentiment analysis
    sid = SentimentIntensityAnalyzer()
    vader_scores_df = pd.DataFrame([vader_sentiment(preprocessed_text, sid)],
                                   columns=['Positive_VADER_Count', 'Negative_VADER_Count'])

    # Concatenate features
    selected_features = pd.concat(
        [vectorized_text_df, pos_neg_conn_counts_df, pos_neg_counts_df, vader_scores_df, lda_features_df], axis=1)

    # Normalize features
    columns_to_normalize = ['Positive_Connotation_Count', 'Negative_Connotation_Count',
                            'Positive_Word_Count', 'Negative_Word_Count',
                            'Positive_VADER_Count', 'Negative_VADER_Count']
    X = normalize_features(selected_features, columns_to_normalize)

    # Scale features
    X.columns = X.columns.astype(str)  # Convert feature names to strings
    X_scaled = scaler.transform(X)

    # Predict sentiment
    prediction = model.predict(X_scaled)[0]
    print(prediction)

    # Map predictions to sentiment labels
    sentiment_label = "Positive" if prediction[0] > 0.5 else "Negative"
    print(sentiment_label)

    return jsonify({'sentiment': sentiment_label})


if __name__ == '__main__':
    app.run(debug=True)
