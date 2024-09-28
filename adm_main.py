import os
import re
import csv
import json
import asyncio
import subprocess
import numpy as np
import pandas as pd
from pydub import AudioSegment
from deepgram_captions import DeepgramConverter
from datasets import load_dataset, Audio, load_from_disk
from datasets.features.features import require_decoding
from datasets.download.streaming_download_manager import xgetsize
from datasets.utils.py_utils import convert_file_size_to_int
from datasets.table import embed_table_storage
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HUGGINGFACE_TOKEN:
    HUGGINGFACE_TOKEN = input("Your HuggingFace Token env variable isn't set. It's ok, you can provide it here. Enter your Hugging Face access token: ")

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
if not DEEPGRAM_API_KEY:
    DEEPGRAM_API_KEY = input("Your Deepgram API key env variable isn't set. It's ok, you can provide it here. Enter your Deepgram API key: ")


def collect_USER_INPUTS():
    HF_USERNAME = input("Enter your Hugging Face username: ")
    REPO_NAME = input("Enter the repository name: ")
    COMBINED_USERNAME_REPOID = f"{HF_USERNAME}/{REPO_NAME}"
    SKIP_STEP_1_2 = input("Do you want to skip Step 1/2 (Transcribe and convert audio)? (y/n): ").lower() == 'y'
    SPEAKER_NAME = input("Enter the SPEAKER_NAME: ")
    EVAL_PERCENTAGE = float(input("Enter the EVAL_PERCENTAGE (percentage of data to move to evaluation set): "))
    inputs = {
        'HF_USERNAME': HF_USERNAME,
        'REPO_NAME': REPO_NAME,
        'COMBINED_USERNAME_REPOID': COMBINED_USERNAME_REPOID,
        'SKIP_STEP_1_2': SKIP_STEP_1_2,
        'SPEAKER_NAME': SPEAKER_NAME,
        'EVAL_PERCENTAGE': EVAL_PERCENTAGE,
        'HUGGINGFACE_TOKEN': HUGGINGFACE_TOKEN
    }
    return inputs


def create_and_push_dataset(CSV_FILE_PATH, REPO_NAME):
    # Load the dataset from CSV
    dataset = load_dataset('csv', data_files={"train": CSV_FILE_PATH}, delimiter="|")
    
    # Cast the audio column to Audio type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=44100))
    dataset.push_to_hub(REPO_NAME, private=True)
    print(f"Dataset successfully pushed to Hugging Face Hub under {REPO_NAME}.")


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


def save_dataset_to_parquet(dataset_dict, data_dir):
    for split_name, dataset in dataset_dict.items():
        # Not sure if I will need this. I will leave it here for now.
        decodable_columns = [
            k for k, v in dataset.features.items() if require_decoding(v, ignore_decode_attribute=True)
        ]
        dataset_nbytes = dataset._estimate_nbytes()
        max_shard_size = convert_file_size_to_int('500MB')
        num_shards = int(dataset_nbytes / max_shard_size) + 1
        num_shards = max(num_shards, 1)

        shards = (
            dataset.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards)
        )
        def shards_with_embedded_external_files(shards):
            for shard in shards:
                fmt = shard.format
                shard = shard.with_format("arrow")
                shard = shard.map(
                    embed_table_storage,
                    batched=True,
                    batch_size=1000,
                    keep_in_memory=True,
                )
                shard = shard.with_format(**fmt)
                yield shard

        shards = shards_with_embedded_external_files(shards)
        os.makedirs(data_dir, exist_ok=True)

        for index, shard in tqdm(
            enumerate(shards),
            desc=f"Save the dataset shards for split '{split_name}'",
            total=num_shards,
        ):
            shard_path = f"{data_dir}/{split_name}-{index:05d}-of-{num_shards:05d}.parquet"
            shard.to_parquet(shard_path)
        
        print(f"Dataset split '{split_name}' saved as Parquet files in {data_dir}")


def run_initial_processing(COMBINED_USERNAME_REPOID, REPO_NAME):
    dataspeech_main_path = os.path.join(PROJECT_ROOT, "dataspeech", "dataspeech", "main.py")
    env = os.environ.copy()
    command = [
        "python", dataspeech_main_path,
        COMBINED_USERNAME_REPOID,
        "--configuration", "default",
        "--text_column_name", "text",
        "--audio_column_name", "audio",
        "--cpu_num_workers", "8",
        "--repo_id", REPO_NAME,
        "--apply_squim_quality_estimation",
    ]

    try:
        print("Running initial processing and pushing to Hugging Face Hub...")
        subprocess.run(command, check=True, env=env)
        print("Initial processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during initial processing:")


def run_metadata_to_text(COMBINED_USERNAME_REPOID, REPO_NAME, bin_edges_path, text_bins_path, UNFILTERED_PARQUET_DIR):
    metadata_to_text_script_path = os.path.join(
        PROJECT_ROOT, "dataspeech", "scripts", "metadata_to_text.py"
    )
    env = os.environ.copy()
    command = [
        "python", metadata_to_text_script_path,
        COMBINED_USERNAME_REPOID,
        "--repo_id", REPO_NAME,
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
        subprocess.run(command, check=True, env=env)
        print("Metadata to text processing completed successfully.")

        dataset = load_from_disk(UNFILTERED_PARQUET_DIR)
        return dataset

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during metadata to text processing:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return None


def filter_parquet_files(UNFILTERED_PARQUET_DIR):
    try:
        print("Filtering Parquet files...")
        subprocess.run(["python", "filter_parquet.py", UNFILTERED_PARQUET_DIR], check=True)
        print("Filtering completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during filtering:")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")


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


def find_parquet_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

async def main():
    USER_INPUTS = collect_USER_INPUTS()
    AUDIO_DIR_PATH = os.path.join(PROJECT_ROOT, "RAW_AUDIO")
    JSON_DIR_PATH = os.path.join(PROJECT_ROOT, "JSON_DIR_PATH")
    SRT_DIR_PATH = os.path.join(PROJECT_ROOT, "SRTS")
    WAVS_DIR = os.path.join(PROJECT_ROOT, "WAVS")
    PARENT_CSV = os.path.join(PROJECT_ROOT, "PARENT_CSV")
    TRAIN_DIR_PATH = os.path.join(PROJECT_ROOT, "METADATA")
    EVAL_DIR_PATH = os.path.join(PROJECT_ROOT, "METADATA")
    CSV_FILE_PATH = os.path.join(PARENT_CSV, "metadata.csv")
    UNFILTERED_PARQUET_DIR = os.path.join(PROJECT_ROOT, "UNFILTERED_PARQUET")
    FILTERED_PARQUET = os.path.join(PROJECT_ROOT, "FILTERED_PARQUET")
    REPO_NAME = USER_INPUTS['REPO_NAME']
    HF_USERNAME = USER_INPUTS['HF_USERNAME']
    SKIP_STEP_1_2 = USER_INPUTS['SKIP_STEP_1_2']
    SPEAKER_NAME = USER_INPUTS['SPEAKER_NAME']
    EVAL_PERCENTAGE = USER_INPUTS['EVAL_PERCENTAGE']
    COMBINED_USERNAME_REPOID = f"{HF_USERNAME}/{REPO_NAME}"


    # Steps 1 and 2 (transcribe audio and convert JSON to SRT)
    if not SKIP_STEP_1_2:
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
    split_dataset(CSV_FILE_PATH, EVAL_PERCENTAGE, TRAIN_DIR_PATH, EVAL_DIR_PATH)
    
    
    # Push the dataset to Hugging Face Hub
    create_and_push_dataset(CSV_FILE_PATH, REPO_NAME)


    # Step 5: Run initial processing
    run_initial_processing(COMBINED_USERNAME_REPOID, REPO_NAME)


    # Step 6: Run metadata_to_text
    bin_edges_path = os.path.join(PROJECT_ROOT, "dataspeech", "examples", "tags_to_annotations", "v02_bin_edges.json")
    text_bins_path = os.path.join(PROJECT_ROOT, "dataspeech", "examples", "tags_to_annotations", "v02_text_bins.json")
    run_metadata_to_text(COMBINED_USERNAME_REPOID, REPO_NAME, bin_edges_path, text_bins_path, UNFILTERED_PARQUET_DIR)
    dataset = run_metadata_to_text(COMBINED_USERNAME_REPOID, REPO_NAME,  bin_edges_path, text_bins_path, UNFILTERED_PARQUET_DIR)
    if dataset is not None:
        save_dataset_to_parquet(dataset, UNFILTERED_PARQUET_DIR)
    else:
        print("Failed to process metadata to text. Skipping Parquet file creation.")


    # Step 7: Filter the dataset
    try:
        print("Starting Step 7: Filter the dataset")
        filter_parquet_files(UNFILTERED_PARQUET_DIR)
        print("Step 7 completed: Dataset filtered successfully.")
    except Exception as e:
        print(f"An error occurred in Step 7: {e}")


    # Step 8: Push the filtered dataset to Hugging Face Hub
    parquet_files = find_parquet_files(FILTERED_PARQUET)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {FILTERED_PARQUET}")

    # Create a dictionary mapping split names to file lists (Not using splits but its here if you need it)
    data_files = {}
    for file in parquet_files:
        split_name = os.path.basename(file).split('-')[0]
        if split_name not in data_files:
            data_files[split_name] = []
        data_files[split_name].append(file)

    dataset = load_dataset('parquet', data_files=data_files)
    dataset.push_to_hub(REPO_NAME, private=True)
    
    print(f"Dataset successfully pushed to Hugging Face Hub under {COMBINED_USERNAME_REPOID}.")
    print("Step 8 completed: Filtered dataset pushed to Hugging Face Hub successfully.")


if __name__ == "__main__":
    asyncio.run(main())
