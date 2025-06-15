# Script will convert a directory of .parquet files (data speech format) into a directory of .wav files and a metadata CSV file.

import os
import glob
import pandas as pd
import shutil
from tqdm import tqdm
import hashlib
import wave
import io


def process_parquet_files(input_dir, output_audio_dir, output_csv_dir):
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    parquet_files = glob.glob(os.path.join(input_dir, '**', '*.parquet'), recursive=True)

    if not parquet_files:
        print(f"No .parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} .parquet files")

    all_data = []
    audio_counter = 0
    for parquet_file in tqdm(parquet_files, desc="Processing .parquet files"):
        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():
            audio_data = row['audio']
            text = row['text']
            audio_counter += 1

            # Check if path exists and is a valid file path
            if audio_data['path'] is not None and os.path.isfile(audio_data['path']):
                audio_path = audio_data['path']
                # Create unique filename using counter
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                new_filename = f"{audio_counter:06d}_{base_name}.wav"
                new_audio_path = os.path.join(output_audio_dir, new_filename)
                shutil.copy2(audio_path, new_audio_path)
            else:
                # Generate a unique filename for each segment
                if audio_data['path'] is not None and audio_data['path'].endswith('.wav'):
                    # Use the existing filename base but make it unique
                    base_name = os.path.splitext(os.path.basename(audio_data['path']))[0]
                    filename = f"{audio_counter:06d}_{base_name}.wav"
                else:
                    # Generate a filename with counter
                    filename = f"{audio_counter:06d}_audio.wav"
                
                new_audio_path = os.path.join(output_audio_dir, filename)
                
                # Write the audio bytes to a new WAV file
                audio_bytes = audio_data['bytes']
                with wave.open(new_audio_path, 'wb') as wav_file:
                    with io.BytesIO(audio_bytes) as bytes_io:
                        with wave.open(bytes_io, 'rb') as wav_bytes:
                            wav_file.setparams(wav_bytes.getparams())
                            wav_file.writeframes(wav_bytes.readframes(wav_bytes.getnframes()))

            all_data.append({
                'audio': os.path.relpath(new_audio_path, start=os.path.dirname(output_csv_dir)),
                'text': text
            })
            
    final_df = pd.DataFrame(all_data)

    csv_path = os.path.join(output_csv_dir, 'metadata.csv')
    final_df.to_csv(csv_path, index=False, sep='|')

    print(f"Processed {len(all_data)} audio files")
    print(f"CSV file saved to {csv_path}")

if __name__ == "__main__":
    input_dir = "../Automatic-Audio-Dataset-Maker/FILTERED_PARQUET"
    output_audio_dir = "../Automatic-Audio-Dataset-Maker/FILTERED_PARQUET" + "/FINAL_WAVS"
    output_csv_dir = "../Automatic-Audio-Dataset-Maker/FILTERED_PARQUET" + "/CSV"

    process_parquet_files(input_dir, output_audio_dir, output_csv_dir)
