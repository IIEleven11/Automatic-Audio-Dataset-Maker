import os
from pydub import AudioSegment
import re
import csv

# Edit these as needed
AUDIO_DIR_PATH = "/home/eleven/deepgram/rawaudio"
SRT_DIR_PATH = "/home/eleven/deepgram/converted_SRTs"
OUTPUT_DIR = "/home/eleven/audio/segments"
CSV_FILE_PATH = "/home/eleven/csvs/metadata.csv"

# Speaker name (change this variable as needed)
SPEAKER_NAME = "Im_A_Speaker"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to format the SRT ms
def srt_time_to_ms(srt_time):
    hours, minutes, seconds, milliseconds = map(int, re.split('[:,]', srt_time))
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds

# Prepare CSV file, were using audio_file|text|speaker_name as a header with pipe delimiter
with open(CSV_FILE_PATH, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter='|')
    csv_writer.writerow(['audio_file', 'text', 'speaker_name'])

    srt_files = [f for f in os.listdir(SRT_DIR_PATH) if f.endswith('.srt')]

    for srt_file in srt_files:
        base_name = os.path.splitext(srt_file)[0]
        audio_file = f"{base_name}.wav"
        audio_path = os.path.join(AUDIO_DIR_PATH, audio_file)
        srt_path = os.path.join(SRT_DIR_PATH, srt_file)

        if os.path.exists(audio_path):
            audio = AudioSegment.from_wav(audio_path)

            with open(srt_path, 'r') as srt_file:
                srt_content = srt_file.read()

            segments = re.findall(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', srt_content, re.DOTALL)

            for segment in segments:
                index, start_time, end_time, text = segment
                start_ms = srt_time_to_ms(start_time)
                end_ms = srt_time_to_ms(end_time)

                audio_segment = audio[start_ms:end_ms]

                segment_filename = f"{base_name}_segment_{index}.wav"
                segment_filepath = os.path.join(OUTPUT_DIR, segment_filename)
                audio_segment.export(segment_filepath, format="wav")
                print(f"Segment {index} saved to {segment_filepath}")

                csv_writer.writerow([segment_filename, text, SPEAKER_NAME])
        else:
            print(f"No corresponding audio file found for {srt_file}")

print("All segments have been processed and saved.")
print(f"CSV file has been saved to {CSV_FILE_PATH}")