import subprocess

# List of scripts to run in order
scripts = [
    "src/data_ingestion.py",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/model_training.py",
    "src/model_evaluation.py"
]

for script in scripts:
    print(f"\n🔧 Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"❌ Error in {script}:\n{result.stderr}")
        break  # Stop the pipeline if any script fails
