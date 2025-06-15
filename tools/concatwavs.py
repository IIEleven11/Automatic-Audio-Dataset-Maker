
# This script concatenates all the WAV files in the input directory into a single WAV file in the output directory. 

import os
from pydub import AudioSegment
from tqdm import tqdm
import math
import traceback
import multiprocessing
import tempfile

# Set the input and output directories
input_dir = "../Automatic-Audio-Dataset-Maker/RAW_AUDIO"
output_dir = "../Automatic-Audio-Dataset-Maker/CONCATENATED_AUDIO"


os.makedirs(output_dir, exist_ok=True)


print("Scanning input directory for WAV files...")
wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
print(f"Found {len(wav_files)} WAV files.")

total_files = len(wav_files)
files_per_output = math.ceil(total_files / 20)
print(f"Will process approximately {files_per_output} files per output.")


def process_file(file):
    try:
        audio = AudioSegment.from_wav(os.path.join(input_dir, file))
        return audio
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        traceback.print_exc()
        return None

def concatenate_wav_files(file_group, output_path):
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_file, file_group), total=len(file_group), desc="Processing files"))
    
    combined = AudioSegment.empty()
    for audio in results:
        if audio is not None:
            combined += audio
    
    try:
        combined.export(output_path, format="wav")
        print(f"Successfully exported {output_path}")
    except Exception as e:
        print(f"Error exporting {output_path}: {str(e)}")
        traceback.print_exc()


def process_batch(batch, batch_num):
    output_file = os.path.join(output_dir, f"concatenated_{batch_num}.wav")
    print(f"Processing batch {batch_num}")
    concatenate_wav_files(batch, output_file)


batch_size = 1000  # Adjust this value based on your system
num_batches = math.ceil(total_files / batch_size)

for i in tqdm(range(num_batches), desc="Processing batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_files)
    batch = wav_files[start_idx:end_idx]
    process_batch(batch, i + 1)

print(f"Concatenation complete. {len(os.listdir(output_dir))} files created in {output_dir}")

