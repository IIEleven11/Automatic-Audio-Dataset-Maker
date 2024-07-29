import json
import os
from deepgram_captions import DeepgramConverter
import numpy as np

# Edit these as needed
AUDIO_DIR_PATH = "/home/eleven/deepgram/audio"
JSON_DIR_PATH = "/home/eleven/deepgram/jsons"
SRT_OUTPUT_DIR = "/home/eleven/deepgram/converted_SRTs"


os.makedirs(SRT_OUTPUT_DIR, exist_ok=True)


def gaussian_duration(min_duration, max_duration):
    mean_duration = (min_duration + max_duration) / 2
    std_duration = (max_duration - min_duration) / 4
    duration = np.random.normal(mean_duration, std_duration)
    return max(min_duration, min(duration, max_duration))


def format_time(seconds):
    """Format time in seconds to SRT time format."""
    milliseconds = int((seconds - int(seconds)) * 1000)
    time_str = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{milliseconds:03}"
    return time_str


def generate_srt(captions):
    """Generate SRT formatted string from captions."""
    srt_content = ""
    for i, (start, end, text) in enumerate(captions, 1):
        srt_content += f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
    return srt_content

def process_transcription(json_path):
    with open(json_path, 'r') as f:
        dg_response = json.load(f)
    transcription = DeepgramConverter(dg_response)

    # Get lines from the transcription
    line_length = 250  # Maximum characters per line, edit this as needed
    lines = transcription.get_lines(line_length)

    # Convert lines to the format expected by the srt function
    captions = []
    for line_group in lines:
        for line in line_group:
            start_time = line.get('start')
            end_time = line.get('end')
            text = line.get('punctuated_word')
            if start_time is not None and end_time is not None and text is not None:
                captions.append((start_time, end_time, text))

    # 1.2 to 15 second segment audio length limit, edit this as needed
    processed_captions = []
    current_start = None
    current_end = None
    current_text = ""
    segment_duration = gaussian_duration(1.2, 15) # edit this as needed

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
            segment_duration = gaussian_duration(1.2, 15)  # If you changed the segment duration, recalculate it for the next segment

    # Add the last caption if it meets the criteria
    if current_end - current_start >= 1.2:
        processed_captions.append((current_start, current_end, current_text.strip()))

    return processed_captions

def main():
    audio_files = [f for f in os.listdir(AUDIO_DIR_PATH) if f.endswith('.wav')]
    json_files = [f for f in os.listdir(JSON_DIR_PATH) if f.endswith('.json')]

    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        json_file = f"{base_name}.json"
        json_path = os.path.join(JSON_DIR_PATH, json_file)

        if json_file in json_files:
            processed_captions = process_transcription(json_path)
            if processed_captions:
                srt_content = generate_srt(processed_captions)
                srt_file_path = os.path.join(SRT_OUTPUT_DIR, f"{base_name}.srt")
                with open(srt_file_path, "w") as srt_file:
                    srt_file.write(srt_content)
                print(f"Transcription saved to {srt_file_path}")
            else:
                print(f"No captions were generated for {audio_file}")
        else:
            print(f"No corresponding JSON file found for {audio_file}")

if __name__ == "__main__":
    main()