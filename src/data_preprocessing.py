import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Download NLTK data for stopwords and tokenization
nltk.download('stopwords')
nltk.download('punkt')

# Create a logs folder if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging for the script
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Set up console logging
console_handeler = logging.StreamHandler()
console_handeler.setLevel('DEBUG')

# Set up file logging
log_file_name = os.path.join(log_dir, 'data_preprocessing.log')
file_handeler = logging.FileHandler(log_file_name)
file_handeler.setLevel('DEBUG')

# Define log message format
formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s')
file_handeler.setFormatter(formatter)
console_handeler.setFormatter(formatter)

# Attach loggers
logger.addHandler(console_handeler)
logger.addHandler(file_handeler)


# Function to clean and process a single text input
def transform_text(text):
    try:
        ps = PorterStemmer()

        # Convert all characters to lowercase
        text = text.lower()

        # Split text into individual words
        text = text.split()

        # Remove punctuations and keep only letters/numbers
        text = [''.join(c for c in word if c.isalnum()) for word in text]
        text = [word for word in text if word]  # Remove any empty strings

        # Remove common stopwords like 'the', 'is', 'and', etc.
        text = [word for word in text if word not in stopwords.words('english')]

        # Apply stemming (e.g., "playing" -> "play")
        text = [ps.stem(word) for word in text]

        # Join list back to single string
        text = " ".join(text)

        # Log only the first 50 characters to keep log short
        # logger.debug("Text transformed successfully: %s", text[:50])
        return text

    except Exception as e:
        logger.error("Error occurred while transforming text: %s", e)
        return ""


# Function to process the entire DataFrame (remove duplicates, encode target, transform text)
def process_df(df, text_column='text', target_column='target'):
    try:
        logger.debug("Start Processing the DataFrame")

        # Encode the labels in target column (e.g., spam/ham -> 0/1)
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target Column labeled properly")

        # Remove duplicate rows
        logger.debug("Start removing duplicate rows from the DataFrame")
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed")

        # Apply text transformation function to text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed properly")
        return df

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred during data normalization: %s", e)
        raise


# Main function to read, process, and save transformed data
def main(text_column='text', target_column='target'):
    try:
        # Read training and test data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        # Process the data
        train_processed_data = process_df(train_data, text_column, target_column)
        test_processed_data = process_df(test_data, text_column, target_column)

        # Create a folder to store processed data
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        # Save the processed data to new CSV files
        train_processed_data.to_csv(os.path.join(data_path, "train_processed_data.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"), index=False)

        logger.debug("Processed data saved to %s", data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()