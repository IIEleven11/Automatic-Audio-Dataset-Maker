import os
import re
import csv
import json
import time
import yaml
import torch
import pysrt
import shutil
import asyncio
import whisper
import logging
import datetime
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from deepgram import Deepgram
from pydub import AudioSegment
import tools.constants as constants
from datasets.table import embed_table_storage
from deepgram_captions import DeepgramConverter
from requests.exceptions import RequestException
from tools.normalize_folder import normalize_audio
from datasets.features.features import require_decoding
from datasets.utils.py_utils import convert_file_size_to_int
from datasets.download.streaming_download_manager import xgetsize
from datasets import load_dataset, Audio, load_from_disk, DatasetDict


os.makedirs('logs', exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/adm_main_{timestamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import gc
import torch

gc.enable()


if torch.cuda.is_available():
    torch.cuda.empty_cache()




PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if HUGGINGFACE_TOKEN:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        user = api.whoami(token=HUGGINGFACE_TOKEN)
        logger.info(f"Successfully authenticated as: {user['name']}")
    except Exception as e:
        logger.error(f"Error verifying Hugging Face token: {e}")
        logger.error("Please check your token and try again.")
        HUGGINGFACE_TOKEN = input("Enter your Hugging Face access token: ")


DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
if not DEEPGRAM_API_KEY:
    DEEPGRAM_API_KEY = input("Enter your Deepgram API key: ")

dg_client = Deepgram(DEEPGRAM_API_KEY)
options = {
    "model": "whisper",
    "punctuate": True,
    "utterances": True,
    "paragraphs": True,
    "smart_format": True,
    "filler_words": True
}


def load_yaml_config(yaml_path):
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    # Set default sample rate if not specified
    if 'audio_processing' not in config:
        config['audio_processing'] = {}
    if 'sample_rate' not in config['audio_processing']:
        config['audio_processing']['sample_rate'] = 24000  # Default sample rate
    return config


#MARK: Collect user inputs
def collect_USER_INPUTS(yaml_path=None):
    """Collect user inputs either from YAML file or interactive input."""
    if yaml_path and Path(yaml_path).exists():
        logger.info(f"Loading configuration from {yaml_path}")
        config = load_yaml_config(yaml_path)

        HF_USERNAME = config['huggingface']['username']
        REPO_NAME = config['huggingface']['repo_name']
        COMBINED_USERNAME_REPOID = f"{HF_USERNAME}/{REPO_NAME}"
        SKIP_STEP_1_2 = config['audio_processing']['skip_transcription']

        TRANSCRIPTION_CHOICE = None
        NUM_GPUS = None

        if not SKIP_STEP_1_2:
            TRANSCRIPTION_CHOICE = str(config['audio_processing']['transcription_method'])
            if TRANSCRIPTION_CHOICE == '1':
                available_gpus = torch.cuda.device_count()
                if available_gpus > 0:
                    NUM_GPUS = config['audio_processing']['num_gpus']
                    if NUM_GPUS is None:
                        NUM_GPUS = available_gpus
                    elif not (1 <= NUM_GPUS <= available_gpus):
                        raise ValueError(f"num_gpus must be between 1 and {available_gpus}")
        
        SKIP_DENOISE_NORMALIZE = config['audio_processing']['skip_denoise_normalize']
        SPEAKER_NAME = config['dataset']['speaker_name']
        EVAL_PERCENTAGE = float(config['dataset']['eval_percentage'])


    else:
        # Original interactive input code
        HF_USERNAME = input("Enter your Hugging Face username: ")
        REPO_NAME = input("Enter the repository name: ")
        COMBINED_USERNAME_REPOID = f"{HF_USERNAME}/{REPO_NAME}"
        
        SKIP_STEP_1_2 = input("Do you want to SKIP audio transcription? (y/n): ").lower() == 'y'
        
        TRANSCRIPTION_CHOICE = None
        NUM_GPUS = None
        
        if not SKIP_STEP_1_2:
            while True:
                TRANSCRIPTION_CHOICE = input("Choose transcription method (1 for local Whisper, 2 for Deepgram API): ").strip()
                if TRANSCRIPTION_CHOICE in ['1', '2']:
                    break
                logger.info("Invalid choice. Please enter 1 or 2.")

            if TRANSCRIPTION_CHOICE == '1':
                available_gpus = torch.cuda.device_count()
                if available_gpus > 0:
                    logger.info(f"\nDetected {available_gpus} GPU(s)")
                    while True:
                        gpu_input = input(f"Enter number of GPUs to use (1-{available_gpus}, or press Enter for all): ").strip()
                        if not gpu_input:
                            NUM_GPUS = available_gpus
                            break
                        try:
                            num = int(gpu_input)
                            if 1 <= num <= available_gpus:
                                NUM_GPUS = num
                                break
                            logger.info(f"Please enter a number between 1 and {available_gpus}")
                        except ValueError:
                            logger.error("Please enter a valid number")

        SKIP_DENOISE_NORMALIZE = input("***This functionality is very sensitive, it's likely to cause failure and I suggest skipping it for now.*** Do you want to SKIP denoising and normalizing? (y/n) : ").lower() == 'y'
        
        SPEAKER_NAME = input("Enter the name of the person speaking in the audio: ")
        EVAL_PERCENTAGE = float(input("Enter the percentage of data to move to evaluation set (10-15%): "))

    inputs = {
        'HF_USERNAME': HF_USERNAME,
        'REPO_NAME': REPO_NAME,
        'COMBINED_USERNAME_REPOID': COMBINED_USERNAME_REPOID,
        'TRANSCRIPTION_CHOICE': TRANSCRIPTION_CHOICE,
        'NUM_GPUS': NUM_GPUS,
        'SKIP_STEP_1_2': SKIP_STEP_1_2,
        'SKIP_DENOISE_NORMALIZE': SKIP_DENOISE_NORMALIZE,
        'SPEAKER_NAME': SPEAKER_NAME,
        'EVAL_PERCENTAGE': EVAL_PERCENTAGE,
        'HUGGINGFACE_TOKEN': HUGGINGFACE_TOKEN
    }
    return inputs


#MARK: Transcribe audio
async def transcribe_audio(file_path, dg_client, options, JSON_DIR_PATH, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as audio_file:
                audio_source = {"buffer": audio_file, "mimetype": "audio/wav"}
                response = await dg_client.transcription.prerecorded(audio_source, options)
                logger.info(f"Transcription response for {file_path}:", json.dumps(response, indent=2))

                base_name = os.path.splitext(os.path.basename(file_path))[0]
                json_file_name = f"{base_name}.json"
                json_file_path = os.path.join(JSON_DIR_PATH, json_file_name)

                with open(json_file_path, "w") as json_file:
                    json.dump(response, json_file, indent=2)

                logger.info(f"Transcription saved to {json_file_path}")
                return
        except RequestException as e:
            logger.error(f"Connection error while processing {file_path}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.info(f"Failed to process {file_path} after {max_retries} attempts.")
        except Exception as e:
            logger.error(f"An error occurred while processing {file_path}: {e}")
            break


def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    time_str = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{milliseconds:03}"
    return time_str


#MARK: Transcribe with Whisper
def transcribe_with_whisper(audio_file_path, output_dir, num_gpus=None):
    """
    Transcribe audio using local Whisper model.
    Skip if SRT file already exists.
    """
    # Check if SRT file already exists
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    srt_file_path = os.path.join(output_dir, f"{base_name}.srt")
    
    if os.path.exists(srt_file_path):
        logger.info(f"SRT file already exists for {base_name}, skipping transcription...")
        return True
        
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("turbo").to(device)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        result = model.transcribe(
            audio_file_path,
            language="en",
            word_timestamps=True,
            verbose=True,
            fp16=(device == "cuda") 
        )
        srt_content = ""
        for i, segment in enumerate(result["segments"], 1):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"].strip()
            srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
            
        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
            
        logger.info(f"Transcription saved to {srt_file_path}")
        
        if device == "cuda":
            torch.cuda.empty_cache()

        return True
    except Exception as e:
        logger.error(f"Error transcribing {audio_file_path}: {e}")
        return False


#MARK: Generate SRT




#MARK: Process transcription
def process_transcription(json_path, SRT_DIR_PATH):
    """
    Convert Deepgram JSON response to SRT format.
    """
    try:
        def generate_srt(captions):
            srt_content = ""
            for i, (start, end, text) in enumerate(captions, 1):
                srt_content += f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
            return srt_content
        
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

        srt_content = generate_srt(captions)

        base_name = os.path.splitext(os.path.basename(json_path))[0]
        srt_file_path = os.path.join(SRT_DIR_PATH, f"{base_name}.srt")
        
        with open(srt_file_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
            
        logger.info(f"Created SRT file: {srt_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing transcription for {json_path}: {str(e)}")
        return False


#MARK: Segment audio and create metadata
def segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR_PREDENOISE, PARENT_CSV, SPEAKER_NAME):
    """
    Audio segmentation using Gaussian distribution for segment durations.
    """
    logger.info("Starting audio segmentation and metadata creation...")
    os.makedirs(WAVS_DIR_PREDENOISE, exist_ok=True)
    metadata_entries = []


    def parse_srt(srt_file_path):
        """Parse the .srt file and return a list of subtitles with start and end times in seconds."""
        subtitles = pysrt.open(srt_file_path)
        subs = []
        for sub in subtitles:
            start_time = sub.start.ordinal / 1000.0
            end_time = sub.end.ordinal / 1000.0
            text = sub.text.replace('\n', ' ').strip()
            subs.append({'start': start_time, 'end': end_time, 'text': text})
        return subs


    def generate_gaussian_durations(total_duration, min_length=2, max_length=18):
        """Generate segment durations following a truncated Gaussian distribution."""
        mean = (min_length + max_length) / 2
        std_dev = (max_length - min_length) / 6
        durations = []
        accumulated = 0
        while accumulated < total_duration:
            duration = np.random.normal(mean, std_dev)
            duration = max(min(duration, max_length), min_length)
            
            remaining = total_duration - accumulated
            
            if remaining < min_length:
                if durations:
                    if durations[-1] + remaining <= max_length:
                        durations[-1] += remaining
                break
            if accumulated + duration > total_duration:
                remaining = total_duration - accumulated
                if min_length <= remaining <= max_length:
                    durations.append(remaining)
                elif remaining > max_length:
                    while remaining > 0:
                        if remaining > max_length:
                            durations.append(max_length)
                            remaining -= max_length
                        else:
                            if remaining >= min_length:
                                durations.append(remaining)
                            elif durations:
                                durations[-1] += remaining
                            break
                break
            durations.append(duration)
            accumulated += duration
        return durations


    def adjust_segments(subs, durations):
        """Adjust the segments to match the desired durations."""
        adjusted_segments = []
        i = 0
        num_subs = len(subs)
        start_time = subs[0]['start']
        
        while i < num_subs:
            if not durations:
                break
                
            segment_duration = durations.pop(0)
            target_end_time = start_time + segment_duration
            
            current_segment = {
                'start': start_time,
                'text': '',
                'end': start_time
            }
            
            while i < num_subs:
                current_segment['text'] += ' ' + subs[i]['text']
                current_segment['end'] = min(subs[i]['end'], start_time + 18)  # force max duration
                
                if subs[i]['end'] >= target_end_time or current_segment['end'] - current_segment['start'] >= 18:
                    break
                i += 1
                
            segment_duration = current_segment['end'] - current_segment['start']
            if 2 <= segment_duration <= 18:
                current_segment['text'] = current_segment['text'].strip()
                adjusted_segments.append(current_segment)

            i += 1
            if i < num_subs:
                start_time = subs[i]['start']

        return adjusted_segments

    srt_files = [f for f in os.listdir(SRT_DIR_PATH) if f.endswith('.srt')]
    for srt_file in tqdm(srt_files, desc="Processing audio files"):
        srt_file_path = os.path.join(SRT_DIR_PATH, srt_file)
        base_name = os.path.splitext(srt_file)[0]
        wav_file = base_name + '.wav'
        wav_file_path = os.path.join(AUDIO_DIR_PATH, wav_file)
        
        if not os.path.exists(wav_file_path):
            logger.warning(f'Audio file {wav_file} does not exist. Skipping.')
            continue

        subs = parse_srt(srt_file_path)
        if not subs:
            logger.warning(f'No subtitles found in {srt_file}. Skipping.')
            continue

        audio = AudioSegment.from_wav(wav_file_path)
        total_duration = len(audio) / 1000.0  # Convert to seconds
        durations = generate_gaussian_durations(total_duration)
        adjusted_segments = adjust_segments(subs, durations)

        # Process and export audio segments
        for idx, segment in enumerate(adjusted_segments):
            start_ms = segment['start'] * 1000
            end_ms = segment['end'] * 1000
            audio_segment = audio[start_ms:end_ms]
            output_filename = f"{base_name}_{idx+1}.wav"
            output_path = os.path.join(WAVS_DIR_PREDENOISE, output_filename)
            audio_segment.export(output_path, format="wav")
            
            metadata_entries.append({
                'audio': output_path,
                'text': segment['text'],
                'speaker_name': SPEAKER_NAME
            })
        
        logger.info(f'Processed {wav_file} into {len(adjusted_segments)} segments.')

    os.makedirs(PARENT_CSV, exist_ok=True)
    metadata_df = pd.DataFrame(metadata_entries)
    metadata_df.to_csv(os.path.join(PARENT_CSV, "metadata.csv"), sep='|', index=False)
    logger.info(f'Metadata saved to {os.path.join(PARENT_CSV, "metadata.csv")}')


#MARK: Split dataset
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

    logger.info(f"Moved {num_rows_to_move} rows from {PARENT_CSV} to {eval_file_path}")


#MARK: Save dataset to parquet
def save_dataset_to_parquet(dataset_dict, data_dir, sample_rate):
    for split_name, dataset in dataset_dict.items():
        # Not sure if I will need this. I will leave it here for now.
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        
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
        
        logger.info(f"Dataset split '{split_name}' saved as Parquet files in {data_dir}")


#MARK: Create and push dataset
def create_and_push_dataset(CSV_FILE_PATH, COMBINED_USERNAME_REPOID):
    logger.info("Loading and verifying dataset...")

    df = pd.read_csv(CSV_FILE_PATH, delimiter='|')

    valid_rows = []
    for idx, row in tqdm(df.iterrows(), desc="Verifying audio files", total=len(df)):
        audio_path = row['audio']
        if os.path.exists(audio_path):
            # Check for zero-length audio file
            try:
                audio_data, sample_rate = sf.read(audio_path)
                if len(audio_data) == 0:
                    logger.warning(f"Warning: Zero-length audio file: {audio_path}")
                else:
                    valid_rows.append(row)
            except Exception as e:
                logger.error(f"Error reading audio file {audio_path}: {e}")
        else:
            logger.warning(f"Warning: Audio file not found: {audio_path}")

    valid_df = pd.DataFrame(valid_rows)
    if len(valid_df) == 0:
        raise ValueError("No valid audio files found in the dataset!")

    temp_csv_path = CSV_FILE_PATH.replace('.csv', '_verified.csv')
    valid_df.to_csv(temp_csv_path, sep='|', index=False)

    try:
        logger.info(f"Creating dataset with {len(valid_df)} valid audio files...")
        dataset = DatasetDict.from_csv({"train": temp_csv_path}, delimiter="|")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

        logger.info("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            COMBINED_USERNAME_REPOID,
            private=True,
            token=HUGGINGFACE_TOKEN,
        )
        logger.info(f"Dataset successfully pushed to Hugging Face Hub under {COMBINED_USERNAME_REPOID}.")

    except Exception as e:
        logger.error(f"Error during dataset creation or push: {str(e)}")
        logger.debug("\nDebug information:")
        logger.debug(f"Repository ID: {COMBINED_USERNAME_REPOID}")
        logger.debug(f"Token available: {'Yes' if HUGGINGFACE_TOKEN else 'No'}")
        logger.debug(f"Token length: {len(HUGGINGFACE_TOKEN) if HUGGINGFACE_TOKEN else 'N/A'}")
        raise

    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)


#MARK: Run initial processing
def run_initial_processing(COMBINED_USERNAME_REPOID, REPO_NAME, sample_rate):
    import gc
    print("Running initial processing...")
    dataspeech_dir = os.path.join(PROJECT_ROOT, "dataspeech", "dataspeech", "main.py")
    env = os.environ.copy()
    
    # Add preprocessing to ensure text data is valid
    def preprocess_dataset():
        try:
            dataset = load_dataset(COMBINED_USERNAME_REPOID)
            
            # Function to ensure text is valid string
            def clean_text(example):
                if 'text' in example:
                    # Convert to string if not already and handle None/nan values
                    text = str(example['text']) if example['text'] is not None else ""
                    # Remove any problematic characters and normalize
                    text = text.encode('ascii', 'ignore').decode('ascii')
                    example['text'] = text
                return example
            
            # Apply cleaning to dataset
            cleaned_dataset = dataset.map(
                clean_text,
                desc="Cleaning text data"
            )
            
            # Push cleaned dataset back to hub
            cleaned_dataset.push_to_hub(
                COMBINED_USERNAME_REPOID,
                private=True,
                token=HUGGINGFACE_TOKEN
            )
            return True
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            return False

    # Preprocess the dataset first
    if not preprocess_dataset():
        logger.error("Failed to preprocess dataset")
        return False

    command = [
        "python", dataspeech_dir,
        COMBINED_USERNAME_REPOID,
        "--configuration", "default",
        "--text_column_name", "text",
        "--audio_column_name", "audio",
        "--cpu_num_workers", "1",
        "--repo_id", REPO_NAME,
        "--rename_column",
        "--apply_squim_quality_estimation"
    ]
    
    try:
        logger.info("Running initial dataset processing with DataSpeech...")
        logger.info(f"Using DataSpeech main.py at: {dataspeech_dir}")
        subprocess.run(command, check=True, env=env)
        gc.collect()

        # Load dataset without streaming
        logger.info(f"Loading and resampling dataset to {sample_rate}Hz...")
        dataset = load_dataset(COMBINED_USERNAME_REPOID)
        gc.collect()
        
        # Function to safely resample audio
        def resample_audio(example):
            try:
                audio = example['audio']
                if audio['sampling_rate'] != sample_rate:
                    # Load audio data
                    audio_data = audio['array']
                    orig_sr = audio['sampling_rate']
                    
                    # Resample using librosa
                    import librosa
                    resampled_audio = librosa.resample(
                        y=audio_data, 
                        orig_sr=orig_sr, 
                        target_sr=sample_rate
                    )
                    
                    # Update the audio dictionary
                    example['audio'] = {
                        'array': resampled_audio,
                        'sampling_rate': sample_rate
                    }
                    gc.collect()
                return example
            except Exception as e:
                logger.warning(f"Failed to resample audio: {str(e)}")
                return example

        logger.info("Resampling audio files...")
        try:
            resampled_dataset = DatasetDict()
            for split in dataset:
                logger.info(f"Processing split: {split}")
                resampled_dataset[split] = dataset[split].map(
                    resample_audio,
                    desc=f"Resampling {split} split"
                )

            # Cast the audio column with the new sample rate
            for split in resampled_dataset:
                resampled_dataset[split] = resampled_dataset[split].cast_column(
                    "audio", 
                    Audio(sampling_rate=sample_rate)
                )
            gc.collect()

            # Push the resampled dataset back to the hub
            logger.info("Pushing resampled dataset to hub...")
            resampled_dataset.push_to_hub(
                COMBINED_USERNAME_REPOID,
                private=True,
                token=HUGGINGFACE_TOKEN
            )
            gc.collect()
            
            print("Initial processing completed successfully.")
            logger.info("Initial processing completed successfully.")
            return True

        except Exception as e:
            logger.error(f"An error occurred during dataset processing: {str(e)}")
            logger.error("Try running with fewer processes or smaller batch size")
            gc.collect()
            return False
        
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred during initial processing:")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Output: {e.output if hasattr(e, 'output') else 'No output'}")
        logger.error(f"Error: {e.stderr if hasattr(e, 'stderr') else 'No error output'}")
        gc.collect()
        return False



#MARK: Run metadata to text processing
def run_metadata_to_text(COMBINED_USERNAME_REPOID, REPO_NAME, bin_edges_path, text_bins_path, UNFILTERED_PARQUET_DIR, sample_rate):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    env = os.environ.copy()
    speaker_name = config['dataset']['speaker_name']
    metadata_to_text_script_path = os.path.join(
        PROJECT_ROOT, "dataspeech", "scripts", "metadata_to_text.py"
    )
    command = [
        "python", metadata_to_text_script_path,
        COMBINED_USERNAME_REPOID,
        "--repo_id", REPO_NAME,
        "--configuration", "default", 
        "--cpu_num_workers", "1",
        "--save_bin_edges", bin_edges_path,
        "--avoid_pitch_computation",
        "--apply_squim_quality_estimation",
        "--output_dir", UNFILTERED_PARQUET_DIR,
        "--speaker_id_column_name", speaker_name,
    ]
    try:
        logger.info("Running metadata to text processing...")
        subprocess.run(command, check=True, env=env)
        logger.info("Metadata to text processing completed successfully.")

        dataset = load_from_disk(UNFILTERED_PARQUET_DIR)
        
        # Cast the audio column before pushing to the hub
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        dataset.push_to_hub(COMBINED_USERNAME_REPOID, private=True, token=HUGGINGFACE_TOKEN)

        return dataset

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred during metadata to text processing:")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Output: {e.output if hasattr(e, 'output') else 'No output'}")
        logger.error(f"Error: {e.stderr if hasattr(e, 'stderr') else 'No error output'}")
        return None


#MARK: Filter Parquet files
def filter_parquet_files(UNFILTERED_PARQUET_DIR):
    try:
        logger.info("Filtering Parquet files...")
        result = subprocess.run(["python", "filter_parquet.py", UNFILTERED_PARQUET_DIR], capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        
        if "No rows passed the filters" in result.stdout:
            logger.warning("WARNING: No audio files passed the quality filters.")
            return False
        else:
            logger.info("Filtering completed successfully.")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred during filtering:")
        logger.error(f"Command: {e.cmd}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
        

#MARK: Denoise and normalize audio (shitty function will need to be fixed)
def denoise_and_normalize(input_folder, output_folder, dataset_name):
    from tools.denoiser import denoise_audio

    logger.info("Starting denoising and normalizing process...")
    os.makedirs(output_folder, exist_ok=True)
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    successful_files = []

    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        try:
            input_path = os.path.join(input_folder, wav_file)
            output_path = os.path.join(output_folder, wav_file)
            if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                logger.warning(f"Warning: Empty or missing file {wav_file}. Skipping.")
                continue

            audio_data, sample_rate = sf.read(input_path)
            if len(audio_data) < sample_rate * 0.1:  # Minimum 100ms of audio
                logger.warning(f"Warning: Audio file {wav_file} is too short. Copying without processing.")
                sf.write(output_path, audio_data, sample_rate)
                successful_files.append(wav_file)
                continue

            try:
                denoised_audio = denoise_audio(audio_data, sample_rate)
                sf.write(output_path, denoised_audio, sample_rate)
                successful_files.append(wav_file)
            except Exception as e:
                logger.warning(f"Warning: Error denoising {wav_file}: {str(e)}. Using original audio.")
                sf.write(output_path, audio_data, sample_rate)
                successful_files.append(wav_file)
        except Exception as e:
            logger.error(f"Error processing {wav_file}: {str(e)}")
            continue

    logger.info(f"Denoising completed! Successfully processed {len(successful_files)} out of {len(wav_files)} files.")


    #MARK: Safe (in theory) normalize audio (Note- Communism works in theory too)
    def safe_normalize_audio(input_file, output_file):
        try:
            audio_data, sample_rate = sf.read(input_file)
            if len(audio_data) < sample_rate * 0.1:  # Too short to normalize
                sf.write(output_file, audio_data, sample_rate)
                return True
            return normalize_audio(input_file, output_file)
        except Exception as e:
            logger.error(f"Error normalizing {os.path.basename(input_file)}: {str(e)}")
            # Copy original file if normalization fails
            try:
                if input_file != output_file:
                    audio_data, sample_rate = sf.read(input_file)
                    sf.write(output_file, audio_data, sample_rate)
                return True
            except:
                return False

    normalized_files = []
    for wav_file in tqdm(successful_files, desc="Normalizing"):
        input_file = os.path.join(output_folder, wav_file)
        if safe_normalize_audio(input_file, input_file):  # Normalize in place
            normalized_files.append(wav_file)

    logger.info(f"Processing completed! Successfully processed {len(normalized_files)} out of {len(wav_files)} files.")
    return normalized_files if normalized_files else None


#MARK: Update CSV file paths
def update_csv_file_paths(csv_file_path, old_dir, new_dir):
    with open(csv_file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = [lines[0]]  # Keep the header
    for line in lines[1:]:
        parts = line.split('|')
        old_path = parts[0]
        new_path = old_path.replace(old_dir, new_dir)
        parts[0] = new_path
        updated_lines.append('|'.join(parts))

    with open(csv_file_path, 'w') as file:
        file.writelines(updated_lines)

    logger.info(f"Updated CSV file paths from {old_dir} to {new_dir}")


#MARK:Push to hub with retry
def push_to_hub_with_retry(dataset, repo_id, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            dataset.push_to_hub(
                repo_id,
                private=True,
                token=HUGGINGFACE_TOKEN,
            )
            logger.info(f"Dataset successfully pushed to Hugging Face Hub under {repo_id}.")
            return
        except Exception as e:
            logger.info(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.info("Max retries reached. Failed to push dataset to Hugging Face Hub.")
                logger.debug("\nDebug information:")
                logger.info(f"Repository ID: {repo_id}")
                logger.info(f"Token available: {'Yes' if HUGGINGFACE_TOKEN else 'No'}")
                logger.info(f"Token length: {len(HUGGINGFACE_TOKEN) if HUGGINGFACE_TOKEN else 'N/A'}")
                raise


#MARK: Main function
async def main():
    parser = argparse.ArgumentParser(description='Audio Dataset Maker')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    USER_INPUTS = collect_USER_INPUTS(args.config if args.config else None)
    AUDIO_DIR_PATH = os.path.join(PROJECT_ROOT, "RAW_AUDIO")
    JSON_DIR_PATH = os.path.join(PROJECT_ROOT, "JSON_DIR_PATH")
    SRT_DIR_PATH = os.path.join(PROJECT_ROOT, "SRTS")
    WAVS_DIR_PREDENOISE = os.path.join(PROJECT_ROOT, "WAVS_DIR_PREDENOISE")
    WAVS_DIR_POSTDENOISE = os.path.join(PROJECT_ROOT, "WAVS_DIR_POSTDENOISE")
    PARENT_CSV = os.path.join(PROJECT_ROOT, "PARENT_CSV")
    TRAIN_DIR_PATH = os.path.join(PROJECT_ROOT, "METADATA")
    EVAL_DIR_PATH = os.path.join(PROJECT_ROOT, "METADATA")
    CSV_FILE_PATH = os.path.join(PARENT_CSV, "metadata.csv")
    UNFILTERED_PARQUET_DIR = os.path.join(PROJECT_ROOT, "UNFILTERED_PARQUET")
    FILTERED_PARQUET_DIR = os.path.join(PROJECT_ROOT, "FILTERED_PARQUET")
    FILTERED_PARQUET_AND_AUDIO = os.path.join(PROJECT_ROOT, "FILTERED_PARQUET_AND_AUDIO")
    REPO_NAME = USER_INPUTS['REPO_NAME']
    HF_USERNAME = USER_INPUTS['HF_USERNAME']
    SKIP_STEP_1_2 = USER_INPUTS['SKIP_STEP_1_2']
    SPEAKER_NAME = USER_INPUTS['SPEAKER_NAME']
    EVAL_PERCENTAGE = USER_INPUTS['EVAL_PERCENTAGE']
    COMBINED_USERNAME_REPOID = f"{HF_USERNAME}/{REPO_NAME}"
    constants.PROJECT_ROOT = PROJECT_ROOT
    constants.WAVS_DIR_PREDENOISE = WAVS_DIR_PREDENOISE
    constants.WAVS_DIR_POSTDENOISE = WAVS_DIR_POSTDENOISE
    constants.COMBINED_USERNAME_REPOID = COMBINED_USERNAME_REPOID
    constants.FILTERED_PARQUET_AND_AUDIO = FILTERED_PARQUET_AND_AUDIO
    
    
    config = load_yaml_config(args.config if args.config else None)
    sample_rate = config['audio_processing']['sample_rate']

    
    #MARK: Steps 1 and 2 (transcribe audio and convert JSON to SRT)
    if not SKIP_STEP_1_2:
        logger.info("Starting Step 1: Transcribe audio")
        
        os.makedirs(SRT_DIR_PATH, exist_ok=True)
        
        audio_files = [os.path.join(AUDIO_DIR_PATH, f) for f in os.listdir(AUDIO_DIR_PATH) if f.endswith('.wav')]

        if USER_INPUTS['TRANSCRIPTION_CHOICE'] == '1':
            logger.info("Using local Whisper model for transcription...")
            for audio_file in tqdm(audio_files, desc="Transcribing with Whisper"):
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                srt_file_path = os.path.join(SRT_DIR_PATH, f"{base_name}.srt")
            
                if os.path.exists(srt_file_path):
                    tqdm.write(f"Skipping {base_name} - SRT file already exists")
                    continue

                transcribe_with_whisper(audio_file, SRT_DIR_PATH, num_gpus=USER_INPUTS['NUM_GPUS'])

        else:
            logger.info("Using Deepgram API for transcription...")
            for audio_file in audio_files:
                await transcribe_audio(audio_file, dg_client, options, JSON_DIR_PATH)

            logger.info("Converting JSON responses to SRT format...")
            json_files = [f for f in os.listdir(JSON_DIR_PATH) if f.endswith('.json')]
            
            for json_file in json_files:
                json_path = os.path.join(JSON_DIR_PATH, json_file)
                if process_transcription(json_path, SRT_DIR_PATH):
                    logger.info(f"Successfully processed {json_file}")
                else:
                    logger.warning(f"Failed to process {json_file}")

        logger.info("Step 1 completed: Audio transcription finished")


    #MARK: Step 3: Segment audio and create metadata
    logger.info("Starting Step 3: Segment audio and create metadata")
    segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR_PREDENOISE, PARENT_CSV, SPEAKER_NAME)
    logger.info("Step 3 completed: Audio segmented and metadata created")


    #MARK: Step 4: Split dataset / denoise / normalize / push to hub
    logger.info("Starting Step 4: Split dataset, denoise, and push to hub")
    split_dataset(CSV_FILE_PATH, EVAL_PERCENTAGE, TRAIN_DIR_PATH, EVAL_DIR_PATH)

    if USER_INPUTS['SKIP_DENOISE_NORMALIZE']:
        logger.info("Skipping denoising and normalizing as requested...")

        os.makedirs(WAVS_DIR_POSTDENOISE, exist_ok=True)
        for wav_file in tqdm(os.listdir(WAVS_DIR_PREDENOISE), desc="Copying audio files"):
            if wav_file.endswith('.wav'):
                src = os.path.join(WAVS_DIR_PREDENOISE, wav_file)
                dst = os.path.join(WAVS_DIR_POSTDENOISE, wav_file)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    logger.error(f"Error copying {wav_file}: {str(e)}")
    else:
        processed_files = denoise_and_normalize(WAVS_DIR_PREDENOISE, WAVS_DIR_POSTDENOISE, COMBINED_USERNAME_REPOID)
        if not processed_files:
            logger.warning("Warning: Some files failed during processing. Continuing with successfully processed files...")

        # Update CSV to only include successfully processed files
        df = pd.read_csv(CSV_FILE_PATH, delimiter='|')
        if processed_files:
            df = df[df['audio'].apply(lambda x: os.path.basename(x) in processed_files)]

        if len(df) == 0:
            logger.error("Error: No valid files remaining after processing.")
            return
        df.to_csv(CSV_FILE_PATH, sep='|', index=False)
        logger.info(f"Updated CSV file to include {len(df)} valid entries.")

    update_csv_file_paths(CSV_FILE_PATH, WAVS_DIR_PREDENOISE, WAVS_DIR_POSTDENOISE)
    create_and_push_dataset(CSV_FILE_PATH, COMBINED_USERNAME_REPOID)
    logger.info("Step 4 completed: Dataset split, denoised, normalized, and pushed to the HuggingfaceHub")


    #MARK: Step 5: Run initial processing
    logger.info("Starting Step 5: Run initial processing")
    if not run_initial_processing(COMBINED_USERNAME_REPOID, REPO_NAME, sample_rate):
        logger.error("Failed to complete initial processing. Stopping execution.")
        return


    #MARK: Step 6: Run metadata_to_text
    logger.info("Starting Step 6: Run metadata_to_text")
    bin_edges_path = os.path.join(PROJECT_ROOT, "computed_bin_edges.json")
    text_bins_path = os.path.join(PROJECT_ROOT, "dataspeech", "examples", "tags_to_annotations", "v02_text_bins.json")
    dataset = run_metadata_to_text(COMBINED_USERNAME_REPOID, REPO_NAME, bin_edges_path, 
                                text_bins_path, UNFILTERED_PARQUET_DIR, sample_rate)
    
    if dataset is not None:
        save_dataset_to_parquet(dataset, UNFILTERED_PARQUET_DIR, sample_rate)
        logger.info("Step 6 completed: Metadata processed to text and saved as Parquet files.")
    else:
        logger.error("Failed to process metadata to text. Skipping Parquet file creation.")


    #MARK: Step 7: Filter the dataset
    try:
        logger.info("Starting Step 7: Filter the dataset")
        data_passed_filters = filter_parquet_files(UNFILTERED_PARQUET_DIR)
        if not data_passed_filters:
            logger.error("ERROR: No audio files passed the quality filters.")
            logger.error("Please manually review and improve your audio files, or remove low-quality files.")
            logger.error("Exiting the script.")
            return
        logger.info("Step 7 completed: Dataset filtered successfully.")
    except Exception as e:
        logger.error(f"An error occurred in Step 7: {e}")
        return 


    #MARK: Step 8: Push the filtered dataset to Hugging Face Hub
    logger.info("Starting step 8, pushing filtered dataset to Hugging Face Hub...")
    logger.info(f"Contents of {FILTERED_PARQUET_DIR}:")
    for file in os.listdir(FILTERED_PARQUET_DIR):
        logger.info(file)

    logger.info(f"Loading original dataset from hub: {COMBINED_USERNAME_REPOID}")
    original_dataset = load_dataset(COMBINED_USERNAME_REPOID)

    if os.path.exists(os.path.join(FILTERED_PARQUET_DIR, "dataset.parquet")):
        filtered_dataset = load_dataset("parquet", data_files=os.path.join(FILTERED_PARQUET_DIR, "dataset.parquet", ))
    elif os.path.exists(os.path.join(FILTERED_PARQUET_DIR, "train")):
        filtered_dataset = load_dataset("parquet", data_dir=FILTERED_PARQUET_DIR)
    else:
        data_files = [os.path.join(FILTERED_PARQUET_DIR, f) for f in os.listdir(FILTERED_PARQUET_DIR) if f.endswith('.parquet')]
        filtered_dataset = load_dataset("parquet", data_files=data_files)

    filtered_indices = set(filtered_dataset['train']['__index_level_0__'])

    # Create a new dataset with only the filtered rows, preserving the audio column
    def filter_fn(example, idx):
        return idx in filtered_indices

    logger.info("Creating filtered dataset with audio...")
    final_dataset = original_dataset.filter(
        filter_fn,
        with_indices=True
    )

    logger.info("Pushing filtered dataset to hub...")
    push_to_hub_with_retry(final_dataset, COMBINED_USERNAME_REPOID)
    logger.info(f"Filtered dataset successfully pushed to Hugging Face Hub under {COMBINED_USERNAME_REPOID}.")


if __name__ == "__main__":
    asyncio.run(main())
