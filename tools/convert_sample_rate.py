import os
from pydub import AudioSegment

# Directories
input_dir = "/AudioDatasetMaker/tools/tempfolder"
output_dir = "/AudioDatasetMaker/RAW_AUDIO"

os.makedirs(output_dir, exist_ok=True)

# Supported formats
formats = (".mp3", ".webm", ".flac")


for filename in os.listdir(input_dir):
    if filename.endswith(formats):
        input_file_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_file_path = os.path.join(output_dir, output_filename)

        if filename.endswith('.mp3'):
            audio = AudioSegment.from_mp3(input_file_path)
        elif filename.endswith('.webm'):
            audio = AudioSegment.from_file(input_file_path, format="webm")
        elif filename.endswith('.flac'):
            audio = AudioSegment.from_file(input_file_path, format="flac")

        # Set the desired parameters
        audio = audio.set_frame_rate(44100)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit

        audio.export(output_file_path, format="wav")

print("All audio files have been converted to 16-bit PCM, 44100hz, mono.")
