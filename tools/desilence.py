import os
import subprocess
from pathlib import Path

def process_wav_files(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all wav files from input directory
    wav_files = list(Path(input_dir).glob('*.wav'))
    
    for wav_file in wav_files:
        input_path = str(wav_file)
        output_path = os.path.join(output_dir, wav_file.name)
        
        # Construct the unsilence command
        command = [
            'unsilence',
            '-ao',  # audio only
            '-sl', '-35',  # silence level
            '-stt', '0.5',  # silence time threshold
            '-sit', '0.3',  # short interval threshold
            '-y',  # non-interactive mode
            input_path,
            output_path
        ]
        
        print(f"Processing: {wav_file.name}")
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed: {wav_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {wav_file.name}: {e}")

if __name__ == "__main__":
    input_directory = "/home/eleven/AudioDatasetMaker/RAW_AUDIO"
    output_directory = "/home/eleven/AudioDatasetMaker/RAW_AUDIO/unsilenced"
    
    process_wav_files(input_directory, output_directory)