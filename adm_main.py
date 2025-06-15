import os
import gc
import time
import yaml
import torch
import pysrt
import librosa
import asyncio
import whisper
import logging
import datetime
import argparse
import traceback
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from datasets import Dataset
import tools.constants as constants
from datasets.table import embed_table_storage
from datasets.utils.py_utils import convert_file_size_to_int
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
            # Always use Whisper (no need to ask for choice)
            TRANSCRIPTION_CHOICE = '1'  # Whisper is now the only option
            available_gpus = torch.cuda.device_count()
            if available_gpus > 0:
                if yaml_path and Path(yaml_path).exists():
                    NUM_GPUS = config['audio_processing']['num_gpus']
                    if NUM_GPUS is None:
                        NUM_GPUS = available_gpus
                    elif not (1 <= NUM_GPUS <= available_gpus):
                        raise ValueError(f"num_gpus must be between 1 and {available_gpus}")
                else:
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


    def generate_gaussian_durations(total_duration, min_length=1.5, max_length=10):
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
                current_segment['end'] = min(subs[i]['end'], start_time + 10)  # force max duration
                
                if subs[i]['end'] >= target_end_time or current_segment['end'] - current_segment['start'] >= 10:
                    break
                i += 1
                
            segment_duration = current_segment['end'] - current_segment['start']
            if 1.5 <= segment_duration <= 10:
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


        audio, sr = librosa.load(wav_file_path, sr=None)  # sr=None preserves original sample rate
        total_duration = librosa.get_duration(y=audio, sr=sr)
        durations = generate_gaussian_durations(total_duration)
        adjusted_segments = adjust_segments(subs, durations)


        for idx, segment in enumerate(adjusted_segments):
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            audio_segment = audio[start_sample:end_sample]
            
            # Add silence padding (1000ms)
            silence_samples = int(2.0 * sr)
            padding = np.zeros(silence_samples)
            audio_segment = np.concatenate([audio_segment, padding])
            
            output_filename = f"{base_name}_{idx+1}.wav"
            output_path = os.path.join(WAVS_DIR_PREDENOISE, output_filename)
            sf.write(output_path, audio_segment, sr)
            
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
        '''
        decodable_columns = [
            k for k, v in dataset.features.items() if require_decoding(v, ignore_decode_attribute=True)
        ]'''
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
    dataspeech_dir = os.path.join(PROJECT_ROOT, "dataspeech", "main.py")
    env = os.environ.copy()
    
    def preprocess_dataset():
        try:
            dataset = load_dataset(COMBINED_USERNAME_REPOID, streaming=True)
            
            # Function to ensure text is valid string
            def clean_text(example):
                if 'text' in example:
                    # Convert to string if not already and handle None/nan values
                    text = str(example['text']) if example['text'] is not None else ""
                    # Remove any problematic characters and normalize
                    text = text.encode('ascii', 'ignore').decode('ascii')
                    example['text'] = text
                return example
            
            def process_in_chunks(dataset, chunk_size=1000):
                processed_chunks = []
                current_chunk = []
                
                for i, example in enumerate(dataset['train']):
                    current_chunk.append(clean_text(example))
                    
                    if len(current_chunk) >= chunk_size:
                        processed_chunks.append(current_chunk)
                        current_chunk = []
                        
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                if current_chunk:
                    processed_chunks.append(current_chunk)
                
                return processed_chunks

            # Process the dataset in chunks
            chunks = process_in_chunks(dataset)

            
            processed_dataset = Dataset.from_dict({
                key: [example[key] for chunk in chunks for example in chunk]
                for key in chunks[0][0].keys()
            })
            
            # Push cleaned dataset back to hub
            processed_dataset.push_to_hub(
                COMBINED_USERNAME_REPOID,
                private=True,
                token=HUGGINGFACE_TOKEN
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        "--avoid_pitch_computation",
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
                # First check if audio exists and has the expected structure
                if 'audio' not in example:
                    logger.warning("No audio field found in example")
                    return example
                
                audio = example['audio']
                if not isinstance(audio, dict) or 'array' not in audio or 'sampling_rate' not in audio:
                    logger.warning("Audio field has unexpected structure")
                    return example
                
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
        logger.error("An error occurred during initial processing:")
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
        # "--apply_squim_quality_estimation",
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
        logger.error("An error occurred during metadata to text processing:")
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
        logger.error("An error occurred during filtering:")
        logger.error(f"Command: {e.cmd}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False

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

#MARK: Get filtered indices
def get_filtered_indices(filtered_parquet_dir):
    """Get the indices of filtered data from parquet files."""
    if os.path.exists(os.path.join(filtered_parquet_dir, "dataset.parquet")):
        filtered_dataset = load_dataset("parquet", data_files=os.path.join(filtered_parquet_dir, "dataset.parquet"))
    elif os.path.exists(os.path.join(filtered_parquet_dir, "train")):
        filtered_dataset = load_dataset("parquet", data_dir=filtered_parquet_dir)
    else:
        data_files = [os.path.join(filtered_parquet_dir, f) for f in os.listdir(filtered_parquet_dir) if f.endswith('.parquet')]
        filtered_dataset = load_dataset("parquet", data_files=data_files)
    
    return set(filtered_dataset['train']['__index_level_0__'])

#MARK: Create filtered dataset
def create_filtered_dataset(original_dataset, filtered_indices):
    """Create a new dataset containing only the filtered indices."""
    def filter_fn(example, idx):
        return idx in filtered_indices

    return original_dataset.filter(
        filter_fn,
        with_indices=True
    )
    

#MARK: Main function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    return parser.parse_args()

async def main():
    args = parse_args()
    config = load_yaml_config(args.config if args.config else None)
    
    if config['audio_processing'].get('refilter', False):
        logger.info("Refiltering mode activated - skipping to filtering step...")
        try:
            HF_USERNAME = config['huggingface']['username']
            REPO_NAME = config['huggingface']['repo_name']
            COMBINED_USERNAME_REPOID = f"{HF_USERNAME}/{REPO_NAME}"
            UNFILTERED_PARQUET_DIR = os.path.join(PROJECT_ROOT, "UNFILTERED_PARQUET")
            FILTERED_PARQUET_DIR = os.path.join(PROJECT_ROOT, "FILTERED_PARQUET")
            
            sample_rate = config['audio_processing']['sample_rate']
            logger.info("Starting filtering process with updated metrics...")
            data_passed_filters = filter_parquet_files(UNFILTERED_PARQUET_DIR)
            
            if not data_passed_filters:
                logger.error("No audio files passed the quality filters.")
                logger.error("Please adjust the filtering thresholds in filter_parquet.py")
                return
            
            logger.info("Loading original dataset...")
            original_dataset = load_dataset(COMBINED_USERNAME_REPOID)
            
            logger.info("Creating filtered dataset...")
            filtered_indices = get_filtered_indices(FILTERED_PARQUET_DIR)
            final_dataset = create_filtered_dataset(original_dataset, filtered_indices)
            
            logger.info("Pushing filtered dataset to hub...")
            push_to_hub_with_retry(final_dataset, COMBINED_USERNAME_REPOID)
            logger.info(f"Filtered dataset successfully pushed to {COMBINED_USERNAME_REPOID}")
            
            return
            
        except Exception as e:
            logger.error(f"Error during refiltering: {str(e)}")
            raise
    

    logger.info("Starting full dataset processing...")
    parser = argparse.ArgumentParser(description='Audio Dataset Maker')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    USER_INPUTS = collect_USER_INPUTS(args.config if args.config else None)
    AUDIO_DIR_PATH = os.path.join(PROJECT_ROOT, "RAW_AUDIO")
    SRT_DIR_PATH = os.path.join(PROJECT_ROOT, "SRTS")
    WAVS_DIR = os.path.join(PROJECT_ROOT, "WAVS")
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
    constants.WAVS_DIR = WAVS_DIR  # Updated constant name
    constants.COMBINED_USERNAME_REPOID = COMBINED_USERNAME_REPOID
    constants.FILTERED_PARQUET_AND_AUDIO = FILTERED_PARQUET_AND_AUDIO
    
    
    config = load_yaml_config(args.config if args.config else None)
    sample_rate = config['audio_processing']['sample_rate']

    
    #MARK: Steps 1 and 2 (transcribe audio and convert JSON to SRT)
    if not SKIP_STEP_1_2:
        logger.info("Starting Step 1: Transcribe audio")
        
        os.makedirs(SRT_DIR_PATH, exist_ok=True)
        
        audio_files = [os.path.join(AUDIO_DIR_PATH, f) for f in os.listdir(AUDIO_DIR_PATH) if f.endswith('.wav')]

        logger.info("Using Whisper model for transcription...")
        for audio_file in tqdm(audio_files, desc="Transcribing with Whisper"):
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            srt_file_path = os.path.join(SRT_DIR_PATH, f"{base_name}.srt")
        
            if os.path.exists(srt_file_path):
                tqdm.write(f"Skipping {base_name} - SRT file already exists")
                continue

            transcribe_with_whisper(audio_file, SRT_DIR_PATH, num_gpus=USER_INPUTS['NUM_GPUS'])

        logger.info("Step 1 completed: Audio transcription finished")


    #MARK: Step 3: Segment audio and create metadata
    logger.info("Starting Step 3: Segment audio and create metadata")
    segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR, PARENT_CSV, SPEAKER_NAME)
    logger.info("Step 3 completed: Audio segmented and metadata created")

    #MARK: Step 4: Split dataset and push to hub
    logger.info("Starting Step 4: Split dataset and push to hub")
    split_dataset(CSV_FILE_PATH, EVAL_PERCENTAGE, TRAIN_DIR_PATH, EVAL_DIR_PATH)
    create_and_push_dataset(CSV_FILE_PATH, COMBINED_USERNAME_REPOID)
    logger.info("Step 4 completed: Dataset split and pushed to the HuggingfaceHub")


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
    logger.info(f"Filtered dataset saved to {FILTERED_PARQUET_DIR} successfully pushed to Hugging Face Hub under {COMBINED_USERNAME_REPOID}.")


if __name__ == "__main__":
    asyncio.run(main())