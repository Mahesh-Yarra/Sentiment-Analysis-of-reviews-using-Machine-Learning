import zipfile

import pandas as pd

from feature_engineering import extract_features
from neural_network import train_neural_network
from preprocessing import preprocess

# Specify the path to the ZIP file
zip_file_path = r"./Datasets/imdb-dataset-of-50k-movie-reviews.zip"

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Find the first file with a .csv extension (assuming it's the one you want)
    csv_file = [name for name in zip_ref.namelist() if name.endswith('.csv')][0]

    # Read the CSV file directly from the ZIP archive into a DataFrame
    df = pd.read_csv(zip_ref.open(csv_file))

# Load only 5000 samples
df_subset = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Map the sentiment values to 1 for positive and 0 for negative
df_subset['sentiment'] = df_subset['sentiment'].map({'positive': 1, 'negative': 0})

# Preprocess the 'review' column
df_subset['review'] = df_subset['review'].apply(preprocess)

# Extract features
selected_features = extract_features(df_subset)

# Classification
# Call the train_neural_network function and print the classification report
classification_report = train_neural_network(selected_features, df_subset['sentiment'])
print(classification_report)