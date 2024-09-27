import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

def check_huggingface_cli():
    try:
        os.system("huggingface-cli --version")
    except Exception as e:
        print("Hugging Face CLI is not installed. Please install it using: pip install huggingface_hub")
        exit(1)

def login_to_huggingface():
    print("Please log in to Hugging Face CLI using your token.")
    os.system("huggingface-cli login")

def push_dataset_to_hub():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    PATH_TO_CSV_FILE = os.path.join(project_root, "PARENT_CSV", "metadata.csv")
    WAVS_DIR = os.path.join(project_root, "WAVS")


    if not os.path.exists(PATH_TO_CSV_FILE):
        raise FileNotFoundError(f"Unable to find the file at {PATH_TO_CSV_FILE}. Please ensure the file exists.")

    REPO_ID = input("Enter your HuggingFace Username/RepoID (e.g., IIEleven11/myDataset): ")

    # Read the CSV file using pandas
    df = pd.read_csv(PATH_TO_CSV_FILE, delimiter='|', quoting=3, escapechar='\\')
    print(f"CSV file loaded. Shape: {df.shape}")
    print(df.head())

    # Add full path to audio files

    # Convert pandas DataFrame to datasets Dataset
    df['audio'] = df['audio'].apply(lambda x: os.path.join(WAVS_DIR, x))

    dataset = Dataset.from_pandas(df)

    # Create a DatasetDict
    dataset_dict = DatasetDict({"train": dataset})

    # Cast the audio column to the Audio type

    # Cast the audio column to the Audio type
    dataset_dict = dataset_dict.cast_column("audio", Audio())

    # Push dataset to Hugging Face Hub
    dataset_dict.push_to_hub(REPO_ID, private=True)
    print(f"Dataset successfully pushed to Hugging Face Hub under {REPO_ID}.")

if __name__ == "__main__":
    check_huggingface_cli()
    login_to_huggingface()
    push_dataset_to_hub()
