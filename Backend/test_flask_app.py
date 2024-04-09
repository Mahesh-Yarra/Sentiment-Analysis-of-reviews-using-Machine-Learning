import requests

# URL of your Flask app's /predict endpoint
url = 'http://127.0.0.1:5000/predict'

# List of texts to test
texts = [
    "This is a great movie!"
]

# Send POST requests with each text and print the predicted sentiment
for text in texts:
    # Create JSON payload
    payload = {'text': text}

    # Send POST request
    response = requests.post(url, json=payload)

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Get the sentiment prediction from the response JSON
        sentiment = response.json()['sentiment']

        # Print the text and predicted sentiment
        print(f"Text: '{text}'")
        print(f"Predicted Sentiment: {sentiment}")
        print()
    else:
        print(f"Error: Failed to predict sentiment for text '{text}'")
