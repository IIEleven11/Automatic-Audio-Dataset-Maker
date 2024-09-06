import os
import asyncio
import json
import numpy as np
import pandas as pd
from pydub import AudioSegment
import re
import csv
from deepgram import Deepgram
from deepgram_captions import DeepgramConverter

# Function to transcribe audio files
async def transcribe_audio(file_path, dg_client, options, JSON_DIR_PATH):
    try:
        with open(file_path, "rb") as audio_file:
            audio_source = {"buffer": audio_file, "mimetype": "audio/wav"}
            response = await dg_client.transcription.prerecorded(audio_source, options)
            print(f"Transcription response for {file_path}:", json.dumps(response, indent=2))

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            json_file_name = f"{base_name}.json"
            json_file_path = os.path.join(JSON_DIR_PATH, json_file_name)

            with open(json_file_path, "w") as json_file:
                json.dump(response, json_file, indent=2)

            print(f"Transcription saved to {json_file_path}")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

# Function to convert JSON to SRT
def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    time_str = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{milliseconds:03}"
    return time_str

def generate_srt(captions):
    srt_content = ""
    for i, (start, end, text) in enumerate(captions, 1):
        srt_content += f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
    return srt_content

def process_transcription(json_path, SRT_DIR_PATH):
    with open(json_path, 'r') as f:
        dg_response = json.load(f)
    transcription = DeepgramConverter(dg_response)

    line_length = 250
    lines = transcription.get_lines(line_length)

    captions = []
    for line_group in lines:
        for line in line_group:
            start_time = line.get('start')
            end_time = line.get('end')
            text = line.get('punctuated_word')
            if start_time is not None and end_time is not None and text is not None:
                captions.append((start_time, end_time, text))

    processed_captions = []
    current_start = None
    current_end = None
    current_text = ""
    segment_duration = np.random.normal(8.1, 3.45)

    for start, end, text in captions:
        if current_start is None:
            current_start = start
            current_end = end
            current_text = text
        elif end - current_start <= segment_duration and len(current_text + " " + text) <= 250:
            current_end = end
            current_text += " " + text
        else:
            if current_end - current_start >= 1.2:
                processed_captions.append((current_start, current_end, current_text.strip()))
            current_start = start
            current_end = end
            current_text = text
            segment_duration = np.random.normal(8.1, 3.45)

    if current_end - current_start >= 1.2:
        processed_captions.append((current_start, current_end, current_text.strip()))

    return processed_captions

# Function to segment audio and create metadata
def srt_time_to_ms(srt_time):
    hours, minutes, seconds, milliseconds = map(int, re.split('[:,]', srt_time))
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds

def segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR, PARENT_CSV, SPEAKER_NAME):
    os.makedirs(WAVS_DIR, exist_ok=True)

    # Automatically name the CSV file as metadata.csv
    CSV_FILE_PATH = os.path.join(PARENT_CSV, "metadata.csv")

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
                    segment_filepath = os.path.join(WAVS_DIR, segment_filename)
                    audio_segment.export(segment_filepath, format="wav")
                    print(f"Segment {index} saved to {segment_filepath}")

                    csv_writer.writerow([segment_filename, text, SPEAKER_NAME])
            else:
                print(f"No corresponding audio file found for {srt_file}")

    print("All segments have been processed and saved.")
    print(f"CSV file has been saved to {CSV_FILE_PATH}")

# Function to split the dataset
def split_dataset(PARENT_CSV, eval_percentage, train_dir_path, eval_dir_path):
    train_df = pd.read_csv(PARENT_CSV, delimiter="|")

    num_rows_to_move = int(len(train_df) * eval_percentage / 100)

    rows_to_move = train_df.sample(n=num_rows_to_move, random_state=42)
    train_df = train_df.drop(rows_to_move.index)
    eval_df = rows_to_move

    # Automatically name the train and eval files
    train_file_path = os.path.join(train_dir_path, "metadata_train.csv")
    eval_file_path = os.path.join(eval_dir_path, "metadata_eval.csv")

    train_df.to_csv(train_file_path, sep="|", index=False)
    eval_df.to_csv(eval_file_path, sep="|", index=False)

    print(f"Moved {num_rows_to_move} rows from {PARENT_CSV} to {eval_file_path}")

# Main function to execute all steps
async def main():
    # Define project root
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Define paths based on project root
    AUDIO_DIR_PATH = os.path.join(project_root, "RAW_AUDIO")
    JSON_DIR_PATH = os.path.join(project_root, "JSON_DIR_PATH")
    SRT_DIR_PATH = os.path.join(project_root, "SRTS")
    WAVS_DIR = os.path.join(project_root, "WAVS")
    PARENT_CSV = os.path.join(project_root, "PARENT_CSV")
    TRAIN_DIR_PATH = os.path.join(project_root, "METADATA")
    EVAL_DIR_PATH = os.path.join(project_root, "METADATA")

    # Step 1: Transcribe audio
    DEEPGRAM_API_KEY = input("Enter your Deepgram API key: ")

    dg_client = Deepgram(DEEPGRAM_API_KEY)
    options = {
        "model": "whisper",
        "punctuate": True,
        "utterances": True,
        "paragraphs": True,
        "smart_format": True,
        "filler_words": True
    }

    audio_files = [os.path.join(AUDIO_DIR_PATH, f) for f in os.listdir(AUDIO_DIR_PATH) if f.endswith('.wav')]
    for audio_file in audio_files:
        await transcribe_audio(audio_file, dg_client, options, JSON_DIR_PATH)

    # Step 2: Convert JSON to SRT
    json_files = [f for f in os.listdir(JSON_DIR_PATH) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(JSON_DIR_PATH, json_file)
        processed_captions = process_transcription(json_path, SRT_DIR_PATH)
        if processed_captions:
            srt_content = generate_srt(processed_captions)
            base_name = os.path.splitext(json_file)[0]
            srt_file_path = os.path.join(SRT_DIR_PATH, f"{base_name}.srt")
            with open(srt_file_path, "w") as srt_file:
                srt_file.write(srt_content)
            print(f"SRT file saved to {srt_file_path}")
        else:
            print(f"No captions were generated for {json_file}")

    # Step 3: Segment audio and create metadata
    SPEAKER_NAME = input("Enter the SPEAKER_NAME: ")

    # Segment audio and create metadata
    segment_audio_and_create_metadata(SRT_DIR_PATH, AUDIO_DIR_PATH, WAVS_DIR, PARENT_CSV, SPEAKER_NAME)

    # Step 4: Split dataset
    PARENT_CSV_PATH = os.path.join(PARENT_CSV, "metadata.csv")
    EVAL_PERCENTAGE = float(input("Enter the EVAL_PERCENTAGE (percentage of data to move to evaluation set): "))

    # Execute the dataset split
    split_dataset(PARENT_CSV_PATH, EVAL_PERCENTAGE, TRAIN_DIR_PATH, EVAL_DIR_PATH)

if __name__ == "__main__":
    asyncio.run(main())
