import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Specify the directory to save the models
MODELS_DIR = "./Models/"


def train_neural_network(selected_features, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a simple neural network model
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)

    print("Neural Network Accuracy on Test Set:", accuracy)

    # Evaluate the model on the test set and make predictions
    y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

    # Save the trained model in HDF5 format
    model.save(os.path.join(MODELS_DIR, 'sentiment_model.h5'))

    # Return the classification report
    return classification_report(y_test, y_pred)
