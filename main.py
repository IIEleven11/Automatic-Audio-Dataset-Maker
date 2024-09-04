
import asyncio
from Step1_Transcribe_Audio import transcribe_audio_main
from Step4_split_metadata import split_dataset

def main():
    train_file_path = input("Enter the path to the training file: ")
    eval_file_path = input("Enter the path to the evaluation file: ")
    eval_percentage = float(input("Enter the evaluation percentage (e.g., 0.2 for 20%): "))

    split_dataset(train_file_path, eval_percentage)
    audio_dir_path = input("Enter the path to the directory containing audio files: ")
    output_dir = input("Enter the path to the directory for saving transcriptions: ")

    asyncio.run(transcribe_audio_main(audio_dir_path, output_dir))

if __name__ == "__main__":
    main()
