import requests

# URL of your Flask app's /predict endpoint
url = 'http://127.0.0.1:5000/predict'

# List of texts to test

#     "This is a great movie!",
#     "I didn't enjoy this film.",
#     "The acting was superb.",
#     "The plot was confusing and hard to follow.",
#     "I highly recommend this movie to everyone!",
#     "How this film could be classified as Drama, I have no idea. If I were John Voight and Mary Steenburgen, I would be trying to erase this from my CV. It was as historically accurate as Xena and Hercules. Abraham and Moses got melded into Noah. Lot, Abraham's nephew, Lot, turns up thousands of years before he would have been born. Canaanites wandered the earth...really? What were the scriptwriters thinking? Was it just ignorance (\"I remember something about Noah and animals, and Lot and Canaanites and all that stuff from Sunday School\") or were they trying to offend the maximum number of people on the planet as possible- from Christians, Jews and Muslims, to historians, archaeologists, geologists, psychologists, linguists ...as a matter of fact, did anyone not get offended? Anyone who had even a modicum of taste would have winced at this one!",


texts = [
    "How this film could be classified as Drama, I have no idea. If I were John Voight and Mary Steenburgen, I would be trying to erase this from my CV. It was as historically accurate as Xena and Hercules. Abraham and Moses got melded into Noah. Lot, Abraham's nephew, Lot, turns up thousands of years before he would have been born. Canaanites wandered the earth...really? What were the scriptwriters thinking? Was it just ignorance (\"I remember something about Noah and animals, and Lot and Canaanites and all that stuff from Sunday School\") or were they trying to offend the maximum number of people on the planet as possible- from Christians, Jews and Muslims, to historians, archaeologists, geologists, psychologists, linguists ...as a matter of fact, did anyone not get offended? Anyone who had even a modicum of taste would have winced at this one!",
    "This is a great movie!",
    "I did not enjoy this film.",
    "The acting was superb.",
    "The plot was confusing and hard to follow.",
    "I highly recommend this movie to everyone!",
    "How this film could be classified as Drama, I have no idea. If I were John Voight and Mary Steenburgen, I would be trying to erase this from my CV. It was as historically accurate as Xena and Hercules. Abraham and Moses got melded into Noah. Lot, Abraham's nephew, Lot, turns up thousands of years before he would have been born. Canaanites wandered the earth...really? What were the scriptwriters thinking? Was it just ignorance (\"I remember something about Noah and animals, and Lot and Canaanites and all that stuff from Sunday School\") or were they trying to offend the maximum number of people on the planet as possible- from Christians, Jews and Muslims, to historians, archaeologists, geologists, psychologists, linguists ...as a matter of fact, did anyone not get offended? Anyone who had even a modicum of taste would have winced at this one!"
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