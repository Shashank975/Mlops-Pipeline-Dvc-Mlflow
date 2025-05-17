import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging



#Ensure that the Log Directory Exits 
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

#Create a Configuration for the Logging (Console & Log File)
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')


console_handeler =  logging.StreamHandler()
console_handeler.setLevel('DEBUG')



log_file_name = os.path.join(log_dir, 'feature_engineering.log')
file_handeler =  logging.FileHandler(log_file_name)
file_handeler.setLevel('DEBUG')


formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s')
file_handeler.setFormatter(formatter)
console_handeler.setFormatter(formatter)



logger.addHandler(console_handeler)
logger.addHandler(file_handeler)

logger.debug("All the Logs for the feature Engineering are Setup!")

<<<<<<< HEAD

 
=======
# Here we loading the data from the Folder name "data/interim" where we have 2 files Train.csv and text.csv we will gonna apply the load_data function on both csv one by one.

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f"Data Columns: {df.columns.tolist()}")
        logger.debug("Data has been loaded Successfully and we also Fill the Gap!")
        return df

    except FileNotFoundError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("There is some unexpected error occurred: %s", e)
        raise



def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        # Ensure required columns are present
        required_columns = ['text', 'target']
        for col in required_columns:
            if col not in train_data.columns or col not in test_data.columns:
                logger.error(f"Missing column '{col}' in the data.")
                raise ValueError(f"Missing column '{col}' in the data.")

        vectorize = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorize.fit_transform(X_train)
        X_test_bow = vectorize.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug("Tfidf is being applied on the data.")
        return train_df, test_df

    except Exception as e:
        logger.error("Unexpected Error Occurred: %s", e)
        return pd.DataFrame(), pd.DataFrame()


def save_data(df:pd.DataFrame,file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index= False)
        logger.debug("data saved to %s",file_path)
    
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data:%s",e)
        raise


    
def main():
    try:
        max_features = 50
        
        base_path = "C:/Users/ShashankChhoker/Desktop/Shashank_work/Mlops-Pipeline-Dvc-Mlflow/data/interim"
        train_data_path = os.path.join(base_path, "train_processed_data.csv")
        test_data_path = os.path.join(base_path, "test_processed_data.csv")

        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        processed_path = os.path.join("C:/Users/ShashankChhoker/Desktop/Shashank_work/Mlops-Pipeline-Dvc-Mlflow/data", "processed")
        save_data(train_df, os.path.join(processed_path, "train_tfidf.csv"))
        save_data(test_df, os.path.join(processed_path, "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
>>>>>>> a411f20eaeef7684c7a5418449b2b3a4fe9b9b04
