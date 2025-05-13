import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier



#Ensure that the Log Directory Exits 
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

#Create a Configuration for the Logging (Console & Log File)
logger = logging.getLogger('Model Training')
logger.setLevel('DEBUG')


console_handeler =  logging.StreamHandler()
console_handeler.setLevel('DEBUG')



log_file_name = os.path.join(log_dir, 'model_training.log')
file_handeler =  logging.FileHandler(log_file_name)
file_handeler.setLevel('DEBUG')


formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s')
file_handeler.setFormatter(formatter)
console_handeler.setFormatter(formatter)



logger.addHandler(console_handeler)
logger.addHandler(file_handeler)

logger.debug("All the Logs for the Model Training are Setup!")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data has been loaded Successfully from : %s",file_path,df.shape)
        return df

    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train:np.ndarray , y_train:np.ndarray) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError ("The Train and Test should have same number of data points ")
        
        
        clf = RandomForestClassifier()
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')

    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        train_data = load_data('./data/processed/train_tfidf.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        clf = train_model(X_train, y_train)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()