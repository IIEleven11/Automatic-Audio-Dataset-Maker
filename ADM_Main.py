import os
import subprocess
import re
import csv
import json
import asyncio
import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi
from deepgram import Deepgram
from pydub import AudioSegment
from push_to_hub import push_dataset_to_hub
from deepgram_captions import DeepgramConverter
from datasets import DatasetDict, Audio


HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HUGGINGFACE_TOKEN:
    HUGGINGFACE_TOKEN = input("Your HuggingFace Token env variable isn't set. It's ok, you can provide it here. Enter your Hugging Face access token: ")

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
if not DEEPGRAM_API_KEY:
    DEEPGRAM_API_KEY = input("Your Deepgram API key env variable isn't set. It's ok, you can provide it here. Enter your Deepgram API key: ")
    
    
project_root = os.path.dirname(os.path.abspath(__file__))
UNFILTERED_PARQUET_DIR = os.path.join(project_root, "UNFILTERED_PARQUET")
os.makedirs(UNFILTERED_PARQUET_DIR, exist_ok=True)

def collect_user_inputs():
    hf_username = input("Enter your Hugging Face username: ")
    repo_name = input("Enter the repository name: ")
    full_repo_id = f"{hf_username}/{repo_name}"
    skip_step1 = input("Skip Step 1 (Transcribe audio)? (y/n): ").lower() == 'y'
    skip_step2 = input("Skip Step 2 (Convert JSON to SRT)? (y/n): ").lower() == 'y'
    SPEAKER_NAME = input("Enter the SPEAKER_NAME: ")
    EVAL_PERCENTAGE = float(input("Enter the EVAL_PERCENTAGE (percentage of data to move to evaluation set): "))

    inputs = {
        'hf_username': hf_username,
        'repo_name': repo_name,
        'full_repo_id': full_repo_id,
        'skip_step1': skip_step1,
        'skip_step2': skip_step2,
        'SPEAKER_NAME': SPEAKER_NAME,
        'EVAL_PERCENTAGE': EVAL_PERCENTAGE,
        'HUGGINGFACE_TOKEN': HUGGINGFACE_TOKEN
    }
    return inputs


def create_and_push_dataset(csv_file_path, repo_id, UNFILTERED_PARQUET_DIR):
    dataset = DatasetDict.from_csv({"train": csv_file_path}, delimiter="|")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=44100))
    
    # Save a local copy in .arrow format
    dataset.save_to_disk(UNFILTERED_PARQUET_DIR)
    print(f"Local copy of dataset saved to {UNFILTERED_PARQUET_DIR} in .arrow format")
    
    # Convert to DataFrame and save as .parquet
    df = dataset['train'].to_pandas()
    parquet_file_path = os.path.join(UNFILTERED_PARQUET_DIR, "dataset.parquet")
    df.to_parquet(parquet_file_path, engine='pyarrow')
    print(f"Local copy of dataset saved to {parquet_file_path} in .parquet format")
    
    # Push to Hugging Face Hub
    dataset.push_to_hub(repo_id, private=True)
    print(f"Dataset successfully pushed to Hugging Face Hub under {repo_id}.")


async def transcribe_audio(file_path, dg_client, options, JSON_DIR_PATH):
    try:
        with open(file_path, "rb") as audio_file:
            audio_source = {"buffer": audio_file, "mimetype": "audio/wav"}
            response = await dg_client.transcription.prerecorded(audio_source, options)
            print(f"Transcription response for {file_path}:", json.dumps(response, indent=2))

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            json_file_name = f"{base_name}.json"
            json_file_path = os.path.join(JSON_DIR_PATH, json_file_name)

            with open(json_file_path, "w") as json_file:
                json.dump(response, json_file, indent=2)

            print(f"Transcription saved to {json_file_path}")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")


# convert JSON to SRT
def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    time_str = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{milliseconds:03}"
    return time_str

def generate_srt(captions):
    srt_content = ""
    for i, (start, end, text) in enumerate(captions, 1):
        srt_content += f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
    return srt_content


def process_transcription(json_path, SRT_DIR_PATH):
    with open(json_path, 'r') as f:
        dg_response = json.load(f)
    transcription = DeepgramConverter(dg_response)
    line_length = 250
    lines = transcription.get_lines(line_length)
    captions = []
    
    for line_group in lines:
        for line in line_group:
            start_time = line.get('start')
            end_time = line.get('end')
            text = line.get('punctuated_word')
            if start_time is not None and end_time is not None and text is not None:
                captions.append((start_time, end_time, text))

    processed_captions = []
    current_start = None
    current_end = None
    current_text = ""
    segment_duration = np.random.normal(8.1, 3.45)

    for start, end, text in captions:
        if current_start is None:
            current_start = start
            current_end = end
            current_text = text
        elif end - current_start <= segment_duration and len(current_text + " " + text) <= 250:
            current_end = end
            current_text += " " + text
        else:
            if current_end - current_start >= 1.2:
                processed_captions.append((current_start, current_end, current_text.strip()))
            current_start = start
            current_end = end
            current_text = text
            segment_duration = np.random.normal(8.1, 3.45)

    if current_end - current_start >= 1.2:
        processed_captions.append((current_start, current_end, current_text.strip()))

    return processed_captions


# segment audio and create metadata
def srt_time_to_ms(srt_time):
    hours, minutes, seconds, milliseconds = map(int, re.split('[:,]', srt_time))
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds


def segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR, PARENT_CSV, SPEAKER_NAME):
    os.makedirs(WAVS_DIR, exist_ok=True)
    CSV_FILE_PATH = os.path.join(PARENT_CSV, "metadata.csv")

    with open(CSV_FILE_PATH, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='|')
        csv_writer.writerow(['audio', 'text', 'speaker_name'])
        srt_files = [f for f in os.listdir(SRT_DIR_PATH) if f.endswith('.srt')]

        for srt_file in srt_files:
            base_name = os.path.splitext(srt_file)[0]
            audio_file = f"{base_name}.wav"
            audio_path = os.path.join(AUDIO_DIR_PATH, audio_file)
            srt_path = os.path.join(SRT_DIR_PATH, srt_file)

            if os.path.exists(audio_path):
                audio = AudioSegment.from_wav(audio_path)

                with open(srt_path, 'r') as srt_file:
                    srt_content = srt_file.read()

                segments = re.findall(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', srt_content, re.DOTALL)

                for segment in segments:
                    index, start_time, end_time, text = segment
                    start_ms = srt_time_to_ms(start_time)
                    end_ms = srt_time_to_ms(end_time)
                    audio_segment = audio[start_ms:end_ms]
                    segment_filename = f"{base_name}_segment_{index}.wav"
                    segment_filepath = os.path.join(WAVS_DIR, segment_filename)
                    audio_segment.export(segment_filepath, format="wav")
                    print(f"Segment {index} saved to {segment_filepath}")

                    # prepend the full path to the audio file
                    full_audio_path = os.path.join(WAVS_DIR, segment_filename)
                    csv_writer.writerow([full_audio_path, text, SPEAKER_NAME])
            else:
                print(f"No corresponding audio file found for {srt_file}")

    print("All segments have been processed and saved.")
    print(f"CSV file has been saved to {CSV_FILE_PATH}")


def split_dataset(PARENT_CSV, eval_percentage, train_dir_path, eval_dir_path):
    train_df = pd.read_csv(PARENT_CSV, delimiter="|")
    num_rows_to_move = int(len(train_df) * eval_percentage / 100)
    rows_to_move = train_df.sample(n=num_rows_to_move, random_state=42)
    train_df = train_df.drop(rows_to_move.index)
    eval_df = rows_to_move
    train_file_path = os.path.join(train_dir_path, "metadata_train.csv")
    eval_file_path = os.path.join(eval_dir_path, "metadata_eval.csv")
    train_df.to_csv(train_file_path, sep="|", index=False)
    eval_df.to_csv(eval_file_path, sep="|", index=False)

    print(f"Moved {num_rows_to_move} rows from {PARENT_CSV} to {eval_file_path}")


def run_initial_processing(full_repo_id, repo_name, HUGGINGFACE_TOKEN):
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataspeech_main_path = os.path.join(project_root, "dataspeech", "dataspeech", "main.py")
    env = os.environ.copy()
    env['HUGGINGFACE_TOKEN'] = HUGGINGFACE_TOKEN
    
    command = [
        "python", dataspeech_main_path,
        full_repo_id,
        "--configuration", "default",
        "--text_column_name", "text",
        "--audio_column_name", "audio",
        "--cpu_num_workers", "8",
        "--repo_id", repo_name,
        "--apply_squim_quality_estimation",
    ]
    
    try:
        print("Running initial processing and pushing to Hugging Face Hub...")
        subprocess.run(command, check=True, env=env)
        print("Initial processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during initial processing:")


def run_metadata_to_text(full_repo_id, repo_id, bin_edges_path, text_bins_path):
    project_root = os.path.dirname(os.path.abspath(__file__))
    metadata_to_text_script_path = os.path.join(project_root, "dataspeech", "scripts", "metadata_to_text.py")
    env = os.environ.copy()
    env['HUGGINGFACE_TOKEN'] = HUGGINGFACE_TOKEN

    command = [
        "python", metadata_to_text_script_path,
        full_repo_id,
        "--repo_id", repo_id,
        "--configuration", "default",
        "--cpu_num_workers", "8",
        "--path_to_bin_edges", bin_edges_path,
        "--path_to_text_bins", text_bins_path,
        "--avoid_pitch_computation",
        "--apply_squim_quality_estimation",
        "--output_dir", UNFILTERED_PARQUET_DIR
    ]
    try:
        print("Running metadata to text processing...")
        subprocess.run(command, check=True)
        print("Metadata to text processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during metadata to text processing:")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")


def filter_parquet_files():
    try:
        print("Filtering Parquet files...")
        subprocess.run(["python", "filter_parquet.py"], check=True)
        print("Filtering completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during filtering:")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")


async def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR_PATH = os.path.join(project_root, "RAW_AUDIO")
    JSON_DIR_PATH = os.path.join(project_root, "JSON_DIR_PATH")
    SRT_DIR_PATH = os.path.join(project_root, "SRTS")
    WAVS_DIR = os.path.join(project_root, "WAVS")
    PARENT_CSV = os.path.join(project_root, "PARENT_CSV")
    TRAIN_DIR_PATH = os.path.join(project_root, "METADATA")
    EVAL_DIR_PATH = os.path.join(project_root, "METADATA")
    user_inputs = collect_user_inputs()
    hf_username = user_inputs['hf_username']
    repo_name = user_inputs['repo_name']
    full_repo_id = user_inputs['full_repo_id']
    skip_step1 = user_inputs['skip_step1']
    skip_step2 = user_inputs['skip_step2']
    SPEAKER_NAME = user_inputs['SPEAKER_NAME']
    EVAL_PERCENTAGE = user_inputs['EVAL_PERCENTAGE']

    # Steps 1 and 2 (transcribe audio and convert JSON to SRT)
    if not skip_step1:
        print("Skipping Step 1: Transcribe audio")
        json_files = [f for f in os.listdir(JSON_DIR_PATH) if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(JSON_DIR_PATH, json_file)
            processed_captions = process_transcription(json_path, SRT_DIR_PATH)
            if processed_captions:
                srt_content = generate_srt(processed_captions)
                base_name = os.path.splitext(json_file)[0]
                srt_file_path = os.path.join(SRT_DIR_PATH, f"{base_name}.srt")
                with open(srt_file_path, "w") as srt_file:
                    srt_file.write(srt_content)
                print(f"SRT file saved to {srt_file_path}")
            else:
                print(f"No captions were generated for {json_file}")


    # Step 3: Segment audio and create metadata
    segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR, PARENT_CSV, SPEAKER_NAME)

    # Step 4: Split dataset
    PARENT_CSV_PATH = os.path.join(PARENT_CSV, "metadata.csv")
    split_dataset(PARENT_CSV_PATH, EVAL_PERCENTAGE, TRAIN_DIR_PATH, EVAL_DIR_PATH)


    create_and_push_dataset(PARENT_CSV_PATH, full_repo_id, UNFILTERED_PARQUET_DIR)
    
    run_initial_processing(user_inputs['full_repo_id'], user_inputs['repo_name'], user_inputs['HUGGINGFACE_TOKEN'])

    # Run metadata to text processing
    bin_edges_path = os.path.join(project_root, "dataspeech", "examples", "tags_to_annotations", "v02_bin_edges.json")
    text_bins_path = os.path.join(project_root, "dataspeech", "examples", "tags_to_annotations", "v02_text_bins.json")
    run_metadata_to_text(full_repo_id, repo_name, bin_edges_path, text_bins_path)

    # Filter the dataset
    filter_parquet_files()

if __name__ == "__main__":
    asyncio.run(main())
