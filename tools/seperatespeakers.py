import os
import wave
import numpy as np
from pydub import AudioSegment
import re
from datetime import datetime
import subprocess

def time_to_milliseconds(time_str):
    """Convert SRT timestamp to milliseconds"""
    time_obj = datetime.strptime(time_str, '%H:%M:%S,%f')
    return (time_obj.hour * 3600000 + 
            time_obj.minute * 60000 + 
            time_obj.second * 1000 + 
            time_obj.microsecond // 1000)

def parse_srt(srt_path):
    """Parse SRT file and return segments for each speaker"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            timestamp_line = lines[1]
            start_time, end_time = timestamp_line.split(' --> ')
            text_line = ' '.join(lines[2:])  # Join all remaining lines

            if "ChangeMe" in text_line:
                segments.append({
                    'speaker': "ChangeMe",
                    'start': time_to_milliseconds(start_time),
                    'end': time_to_milliseconds(end_time)
                })
    
    return segments

def extract_speaker_segments(audio_path, segments, target_speaker, output_path):
    """Extract and concatenate segments for the target speaker"""
    audio = AudioSegment.from_wav(audio_path)
    speaker_segments = []
    
    for segment in segments:
        if segment['speaker'] == target_speaker:
            segment_audio = audio[segment['start']:segment['end']]
            speaker_segments.append(segment_audio)
    
    if speaker_segments:
        # Concatenate all segments
        final_audio = sum(speaker_segments)
        final_audio.export(output_path, format='wav')
        return True
    return False


def extract_number(filename):
    """Extract the number from the filename, handling different patterns"""
    match = re.search(r'(\d+)\.wav$', filename)
    if match:
        return int(match.group(1))
    return 0

def main():
    input_dir = '/AudioDatasetMaker/RAW_AUDIO'
    output_dir = '/AudioDatasetMaker/DIARIZED'

    os.makedirs(output_dir, exist_ok=True)
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    wav_files.sort(key=extract_number)

    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]
        srt_file = base_name + '.srt'
        
        wav_path = os.path.join(input_dir, wav_file)
        srt_path = os.path.join(input_dir, srt_file)
        
        if not os.path.exists(srt_path):
            print(f"No SRT file found for {wav_file}, skipping...")
            continue
        
        segments = parse_srt(srt_path)
        
        print(f"\nProcessing: {wav_file}")
        output_path = os.path.join(output_dir, f"{base_name}_changeme")
        
        print(f"Extracting ChangeMe's segments from {wav_file}...")
        if extract_speaker_segments(wav_path, segments, "ChangeMe", output_path):
            print(f"Successfully created: {output_path}")
        else:
            print(f"No segments found for ChangeMe in {wav_file}")

if __name__ == "__main__":
    main()
