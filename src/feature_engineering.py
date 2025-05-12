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
