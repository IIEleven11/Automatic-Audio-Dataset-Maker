import os
import re
import csv
from pydub import AudioSegment


def srt_time_to_ms(srt_time):
    hours, minutes, seconds, milliseconds = map(int, re.split('[:,]', srt_time))
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds

def segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR, PARENT_CSV, SPEAKER_NAME):
    os.makedirs(WAVS_DIR, exist_ok=True)

    CSV_FILE_PATH = os.path.join(PARENT_CSV, "metadata.csv")

    with open(CSV_FILE_PATH, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='|')
        csv_writer.writerow(['audio', 'text', 'speaker_name'])

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
                    segment_filepath = os.path.join(WAVS_DIR, segment_filename)

                    audio_segment.export(segment_filepath, format="wav")
                    print(f"Segment {index} saved to {segment_filepath}")

                    # Prepend the full path to the WAVS_DIR for the audio file
                    full_audio_path = os.path.abspath(segment_filepath)
                    csv_writer.writerow([full_audio_path, text, SPEAKER_NAME])
            else:
                print(f"No corresponding audio file found for {srt_file}")

    print("All segments have been processed and saved.")
    print(f"CSV file has been saved to {CSV_FILE_PATH}")
