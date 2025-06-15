# Description: Calculate the total duration of all WAV files in a directory

import os
import wave
from datetime import timedelta
from tqdm import tqdm

<<<<<<< HEAD
target = "../Automatic-Audio-Dataset-Maker/FILTERED_PARQUET/FINAL_WAVS" # Change this path if you need to
=======
target = "/AudioDatasetMaker/WAVS_DIR_PREDENOISE" # Change this path if you need to
>>>>>>> 56f0673ad2084b8d03c5402f657231a67a2b75f3


def get_audio_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0


def calculate_total_duration():
    if not os.path.exists(target):
        print(f"Directory not found: {target}")
        return
        
    total_duration = 0
    wav_files = [f for f in os.listdir(target) if f.endswith('.wav')]
    
    print(f"Directory path: {target}")
    print(f"Found {len(wav_files)} WAV files")
    
    for filename in tqdm(wav_files, desc="Calculating total duration"):
        file_path = os.path.join(target, filename)
        duration = get_audio_duration(file_path)
        total_duration += duration

    duration_timedelta = timedelta(seconds=int(total_duration))
    hours = duration_timedelta.seconds // 3600 + duration_timedelta.days * 24
    minutes = (duration_timedelta.seconds % 3600) // 60
    seconds = duration_timedelta.seconds % 60
    
<<<<<<< HEAD
    print("\nTotal audio duration:")
=======
    print(f"\nTotal audio duration:")
>>>>>>> 56f0673ad2084b8d03c5402f657231a67a2b75f3
    print(f"Hours: {hours}")
    print(f"Minutes: {minutes}")
    print(f"Seconds: {seconds}")
    print(f"Total seconds: {total_duration:.2f}")

if __name__ == "__main__":
    calculate_total_duration()