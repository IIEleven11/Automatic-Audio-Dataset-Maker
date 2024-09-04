
import asyncio
from Step1_Transcribe_Audio import transcribe_audio_main
from Step4_split_metadata import split_dataset
from Step2_Convert_JSON2SRT import convert_json_to_srt

def main():
    train_file_path = input("Enter the path to the training file: ")
    eval_file_path = input("Enter the path to the evaluation file: ")
    eval_percentage = float(input("Enter the evaluation percentage (e.g., 0.2 for 20%): "))

    split_dataset(train_file_path, eval_percentage, train_file_path, eval_file_path)
    audio_dir_path = input("Enter the path to the directory containing audio files: ")
    output_dir = input("Enter the path to the directory for saving transcriptions: ")

    asyncio.run(transcribe_audio_main(audio_dir_path, output_dir))

    json_file_path = input("Enter the path to the JSON file to convert to SRT: ")
    srt_output_path = input("Enter the path to save the SRT file: ")
    convert_json_to_srt(json_file_path, srt_output_path)

if __name__ == "__main__":
    main()
