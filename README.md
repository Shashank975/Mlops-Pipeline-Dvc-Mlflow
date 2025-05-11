# Mlops-Pipeline-Dvc-Mlflow
Learning about the "MLOPS".


### 💡 **Author : Shashank Chhoker**

# 🌟 Data Ingestion and Preprocessing Pipeline

This script streamlines the process of data ingestion, preprocessing, text transformation, and dataset splitting for a spam classification project, complete with robust logging for every step. Here's what it does:


---
### 💡 **File Strcuture:**

project/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── interim/
│       ├── train_processed_data.csv
│       └── test_processed_data.csv
│
├── logs/
│   └── data_preprocessing.log
│
├── preprocess.py
└── README.md

---

### ✅ **Logging Configuration:**

* Creates a `logs` directory (if not already present).
* Configures a logger to output to both the console and a log file (`data_ingestion.log`, `data_preprocessing.log`).
* Logging format: `%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s`

---

### 📦 **Data Loading (`load_data`):**

* Accepts a URL to load a CSV file using **pandas**.
* Logs the data loading status for monitoring.
* Catches parsing errors and other exceptions, logging them effectively.

---

### 🛠️ **Data Preprocessing (`preprocess_data` & `process_df`):**

* Removes unnecessary columns: `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`.
* Renames columns: `v1` → `target`, `v2` → `text`.
* Applies text transformation through the `transform_text` function, including:

  * Lowercasing text
  * Removing punctuations
  * Removing stopwords
  * Applying stemming using **NLTK's PorterStemmer**
* Handles missing columns and general exceptions with appropriate logging.

---

### 🛠️ **Text Transformation (`transform_text`):**

* Converts text to lowercase.
* Splits text into words and removes punctuations.
* Removes common stopwords using **NLTK**.
* Applies stemming to reduce words to their root form.
* Logs the transformed text (first 50 characters).

---

### 💾 **Data Saving (`save_data`):**

* Saves the processed training and testing datasets to the specified data path.
* Directory structure:

  * `data/raw/train.csv`
  * `data/raw/test.csv`
  * `data/interim/train_processed_data.csv`
  * `data/interim/test_processed_data.csv`
* Comprehensive logging ensures that data-saving steps are properly tracked.

---

### 🚀 **Pipeline Execution:**

* Loads data from the specified URL.
* Preprocesses the data, removing unnecessary columns and renaming the relevant ones.
* Transforms text data using NLTK's PorterStemmer and stopword removal.
* Splits the data (80% training, 20% testing).
* Saves the resulting datasets in the `./data` directory.

---

### 🔧 **How to Run:**

1. **Install required libraries:**

   ```bash
   pip install pandas scikit-learn nltk
   ```

2. **Download NLTK data (only once):**

   ```bash
   python -m nltk.downloader stopwords punkt
   ```

3. **Run the data ingestion script:**

   ```bash
   python data_ingestion.py
   ```

4. **Run the data preprocessing script:**

   ```bash
   python data_preprocessing.py
   ```

5. **Check logs:**

   * Monitor the console for live updates.
   * Review detailed logs in the `logs` directory.

---

### 🌐 **Dataset Source:**

* The dataset URL is predefined in the script, pointing to a sample spam dataset hosted on GitHub.

---

### 💡 **Notes:**

* The logging system ensures comprehensive tracking of every step, making debugging and monitoring easier.
* The project structure is organized for scalability and future enhancements.

---

### 💡 **Sample Log Output:**

2025-05-12 10:45:01 -- DEBUG -- data_preprocessing -- Start Processing the DataFrame
2025-05-12 10:45:01 -- DEBUG -- data_preprocessing -- Target Column labeled properly
2025-05-12 10:45:02 -- DEBUG -- data_preprocessing -- Duplicates removed
2025-05-12 10:45:02 -- DEBUG -- data_preprocessing -- Text column transformed properly

---

🚀 Happy Coding!




