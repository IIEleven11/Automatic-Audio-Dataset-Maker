import os
import re
from pathlib import Path

def sanitize_filename(filename):
    # Remove special characters and spaces, keep alphanumeric, underscore, hyphen and dot
    base, ext = os.path.splitext(filename)
    # Replace spaces with underscores and remove/replace special characters
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', base)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized + ext.lower()

def main():
    raw_audio_dir = "../SPEAKERS"
    
    audio = list(Path(raw_audio_dir).glob('*.mp3'))
    
    for audio_path in audio:
        old_name = audio_path.name
        new_name = sanitize_filename(old_name)
        
        if old_name != new_name:
            old_path = audio_path
            new_path = audio_path.parent / new_name
            
            try:
                old_path.rename(new_path)
                print(f"Renamed: {old_name} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {old_name}: {str(e)}")

if __name__ == "__main__":
    main()