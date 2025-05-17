import pickle
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logger = logging.getLogger('spam_detection_test')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Load model
def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        raise

def preprocess_input(texts):
    """ Preprocess the input text, assuming TF-IDF vectorization was used. """
    # You should ensure this vectorizer matches the parameters used during training
    vectorizer = TfidfVectorizer(max_features=5000)  # Match the parameters you used for training
    return vectorizer.transform(texts)

def main():
    model_path = 'models/model.pkl'

    logger.info("Loading model...")
    model = load_pickle(model_path)

    logger.info("Model loaded successfully.")

    while True:
        user_input = input("\n🧪 Enter a message to test (or type 'exit' to quit):\n> ")
        if user_input.lower() in ['exit', 'quit']:
            break

        try:
            # Preprocess the input text (apply the same transformations as during training)
            X_input = preprocess_input([user_input])
            prediction = model.predict(X_input)[0]  # Predict using the trained model
            
            # Output the prediction result
            if prediction == 1:
                print(f"🚨 This message is SPAM!")
            else:
                print(f"✅ This message is HAM (Not Spam).")
        except Exception as e:
            logger.error("Prediction failed: %s", e)

if __name__ == '__main__':
    main()
