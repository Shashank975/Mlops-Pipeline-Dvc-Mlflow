# Mlops-Pipeline-Dvc-Mlflow
Learning about the "MLOPS".

# 🌟 Data Ingestion and Preprocessing Pipeline

This script streamlines the process of data ingestion, preprocessing, and dataset splitting for a spam classification project, complete with robust logging for every step. Here's what it does:

---

### ✅ **Logging Configuration:**

* Creates a `logs` directory (if not already present).
* Configures a logger to output to both the console and a log file (`data_ingestion.log`).
* Logging format: `%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s`

---

### 📦 **Data Loading (`load_data`):**

* Accepts a URL to load a CSV file using **pandas**.
* Logs the data loading status for monitoring.
* Catches parsing errors and other exceptions, logging them effectively.

---

### 🛠️ **Data Preprocessing (`preprocess_data`):**

* Removes unnecessary columns: `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`.
* Renames columns: `v1` → `target`, `v2` → `text`.
* Logs each step of the preprocessing process.
* Handles missing columns and general exceptions with appropriate logging.

---

### 💾 **Data Saving (`save_data`):**

* Saves the processed training and testing datasets to the specified data path.
* Directory structure:

  * `data/raw/train.csv`
  * `data/raw/test.csv`
* Comprehensive logging ensures that data-saving steps are properly tracked.

---

### 🚀 **Pipeline Execution:**

* Loads data from the specified URL.
* Preprocesses the data, removing unnecessary columns and renaming the relevant ones.
* Splits the data (80% training, 20% testing).
* Saves the resulting datasets in the `./data` directory.

---

### 🔧 **How to Run:**

1. **Install required libraries:**

   ```bash
   pip install pandas scikit-learn
   ```

2. **Run the script:**

   ```bash
   python data_ingestion.py
   ```

3. **Check logs:**

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

🚀 Happy Coding!

