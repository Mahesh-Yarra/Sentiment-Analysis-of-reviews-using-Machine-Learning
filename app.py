from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS module
import joblib
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from feature_engineering import update_counts, update_word_counts, vader_sentiment, normalize_features, scale_features
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Paths

# Paths for files
LEXICON_POSITIVE = r"./Datasets/positive-words.txt"
LEXICON_NEGATIVE = r"./Datasets/negative-words.txt"
LEXICON_CONNOTATION = r"./Datasets/connotations.csv"

# Load the trained model and necessary preprocessing steps
# Load the saved model
model = load_model('./Models/sentiment_model.h5')
vectorizer = joblib.load('./Models/tfidf_vectorizer.pkl')
selector = joblib.load('./Models/correlated_feature_picker.pkl')
lda_pipeline = joblib.load('./Models/lda_topics_extractor.pkl')

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

    # Iterate over keys in the JSON data to find the input text
    # for key, value in data.items():
    #     if isinstance(value, str):
    #         input_text = value
    #         break
    #
    # if input_text is None:
    #     # If input text is not found in the JSON data, return an error response
    #     return jsonify({'error': 'Input text not found in request'}), 400

    # Preprocess input text
    # (Apply the same preprocessing steps used during training)
    # Vectorize preprocessed text
    vectorized_text = vectorizer.transform([input_text])
    vectorized_text_df = pd.DataFrame(vectorized_text.toarray(), columns=vectorizer.get_feature_names_out())
    print(f"vectorized_text_df : {type(vectorized_text_df)}")

    # Feature Selection

    # Transform text using LDA pipeline
    lda_features = lda_pipeline.transform([input_text])
    # Convert the array into a DataFrame
    lda_features_df = pd.DataFrame(lda_features)
    print(f"lda_features : {type(lda_features)}")

    # Extract additional features
    pos_neg_conn_counts_df = pd.DataFrame([update_counts(input_text, word_emotion_map)],
                                          columns=['Positive_Connotation_Count', 'Negative_Connotation_Count'])
    print(f"pos_neg_conn_counts_df : {type(pos_neg_conn_counts_df)}")

    # Appending positive and negative word's count
    pos_neg_counts_df = pd.DataFrame([update_word_counts(input_text, positive_words, negative_words)],
                                     columns=['Positive_Word_Count', 'Negative_Word_Count'])
    print(f"pos_neg_counts_df : {type(pos_neg_counts_df)}")

    # Appending VADER features
    # Use VADER for sentiment analysis
    sid = SentimentIntensityAnalyzer()
    vader_scores_df = pd.DataFrame([vader_sentiment(input_text, sid)],
                                   columns=['Positive_VADER_Count', 'Negative_VADER_Count'])
    print(f"vader_scores_df : {type(vader_scores_df)}")

    # Concatenate features
    selected_features = pd.concat(
        [vectorized_text_df, lda_features_df, pos_neg_conn_counts_df, pos_neg_counts_df, vader_scores_df], axis=1)

    # Normalize features
    columns_to_normalize = ['Positive_Connotation_Count', 'Negative_Connotation_Count',
                            'Positive_Word_Count', 'Negative_Word_Count',
                            'Positive_VADER_Count', 'Negative_VADER_Count']
    X = normalize_features(selected_features, columns_to_normalize)

    # Scale features
    X.columns = X.columns.astype(str)  # Convert feature names to strings
    X_scaled = scale_features(X)

    # Predict sentiment
    prediction = model.predict(X_scaled)[0]

    # Return prediction result
    return jsonify({'sentiment': prediction})


if __name__ == '__main__':
    app.run(debug=True)
