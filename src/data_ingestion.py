import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import logging


#Ensure that the Log Directory Exits 
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

#Create a Configuration for the Logging (Console & Log File)
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')


console_handeler =  logging.StreamHandler()
console_handeler.setLevel('DEBUG')



log_file_name = os.path.join(log_dir, 'data_ingestion.log')
file_handeler =  logging.FileHandler(log_file_name)
file_handeler.setLevel('DEBUG')


formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s')
file_handeler.setFormatter(formatter)
console_handeler.setFormatter(formatter)



logger.addHandler(console_handeler)
logger.addHandler(file_handeler)


#Important function for the Loading the Data From the Source :

def load_data(data_url:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data Loaded from the %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV File %s",e)
        raise
    except Exception as e :
        logger.error("Unexcepted Error %s",e)



#Important Function for the cleaning the data:

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)
        logger.debug("Unnecessary Columns are removed")
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug("Columns are Renamed")
        return df
    except KeyError as e:
        logger.error('Missing column in the DataFrame : %s',e)
        raise
    except Exception as e:
        logger.error("Unexcepted Error %s",e)
        raise



def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and Test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise




# Main function
def main():
    data_url = "https://raw.githubusercontent.com/Shashank975/Mlops-Pipeline-Dvc-Mlflow/refs/heads/main/Experiments/spam.csv"
    df = load_data(data_url)
    final_df = preprocess_data(df)
    train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=2)
    save_data(train_data, test_data, data_path='./data')

if __name__ == "__main__":
    main()