#read the file name

import os
import subprocess

# Path to the folder containing audio files
audio_folder = "path/to/folder"
script_path = "path/to/whisper-diarization/diarize_parallel.py"

for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"): 
        audio_file_path = os.path.join(audio_folder, file_name)
        try:
            subprocess.run(
                ["python", script_path, "-a", audio_file_path],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_name}: {e}")
