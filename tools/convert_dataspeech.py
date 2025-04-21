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
    for parquet_file in tqdm(parquet_files, desc="Processing .parquet files"):
        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():
            audio_data = row['audio']
            text = row['text']

            if audio_data['path'] is None:
                # Generate a filename based on the hash of the audio bytes
                audio_bytes = audio_data['bytes']
                filename = hashlib.md5(audio_bytes).hexdigest() + '.wav'
                new_audio_path = os.path.join(output_audio_dir, filename)
                
                # Write the audio bytes to a new WAV file
                with wave.open(new_audio_path, 'wb') as wav_file:
                    with io.BytesIO(audio_bytes) as bytes_io:
                        with wave.open(bytes_io, 'rb') as wav_bytes:
                            wav_file.setparams(wav_bytes.getparams())
                            wav_file.writeframes(wav_bytes.readframes(wav_bytes.getnframes()))
            else:
                audio_path = audio_data['path']
                new_filename = os.path.basename(audio_path)
                new_audio_path = os.path.join(output_audio_dir, new_filename)
                shutil.copy2(audio_path, new_audio_path)

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'FILTERED_PARQUET')
    output_audio_dir = os.path.join(script_dir, 'FILTERED_PARQUET', 'FINAL_WAVS')
    output_csv_dir = os.path.join(script_dir, 'FILTERED_PARQUET', 'CSV')

    process_parquet_files(input_dir, output_audio_dir, output_csv_dir)