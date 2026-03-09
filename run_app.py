import subprocess
import os

DATA_FILE = "data/cleaned/peserta.csv"

if not os.path.exists(DATA_FILE):
    print("Pulling data from DVC...")
    subprocess.run(["dvc", "pull"], check=True)

print("Starting Streamlit...")

subprocess.run([
    "streamlit",
    "run",
    "dashboard/app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0"
])