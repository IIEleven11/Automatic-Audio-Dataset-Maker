# Converts all audio/media files in a directory to 24k mono 16-bit PCM .wav files.

import os
import subprocess

input_dir = "Automatic-Audio-Dataset-Maker/RAW_AUDIO" # Change if needed
output_dir = os.path.join(input_dir, "converted")

os.makedirs(output_dir, exist_ok=True)

media_extensions = {
    '.wav', '.mp3', '.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', 
    '.m4a', '.aac', '.ogg', '.flac', '.wma', '.3gp', '.webm', '.m4v'
}


for filename in os.listdir(input_dir):
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext in media_extensions:
        input_path = os.path.join(input_dir, filename)
        # Change output extension to .wav
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_dir, output_filename)
        
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", input_path,
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ac", "1",              # Mono
            "-ar", "24000",          # 24050 Hz
            output_path
        ]
        
        print(f"Converting {filename} to {output_filename}...")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"✓ Successfully converted {filename}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error converting {filename}: {e.stderr}")

print("Conversion complete.")
