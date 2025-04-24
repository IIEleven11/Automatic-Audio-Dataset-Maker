# once you have srt's and know which speaker you want run this to seperate

import argparse
import os
import re
from pydub import AudioSegment

def time_to_ms(time_str):
    """Convert SRT timestamp format (HH:MM:SS,mmm) to milliseconds"""
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)

def get_speakers_from_srt(srt_file):
    """Get all unique speaker IDs from an SRT file"""
    speakers = set()
    speaker_pattern = re.compile(r'Speaker (\d+):')
    
    with open(srt_file, 'r', encoding='utf-8') as file:
        for line in file:
            match = speaker_pattern.match(line.strip())
            if match:
                speakers.add(match.group(1))
                
    return sorted(list(speakers))

def parse_srt(srt_file, target_speaker, padding_ms=500):
    """Parse SRT file and extract timestamps for the target speaker with padding"""
    segments = []
    
    with open(srt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if line is a number (segment identifier)
        if line.isdigit():

            if i + 1 < len(lines):
                timestamp_line = lines[i+1].strip()
                timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)
                    

                    if i + 2 < len(lines):
                        speaker_line = lines[i+2].strip()
                        speaker_match = re.match(r'Speaker (\d+):', speaker_line)
                        
                        if speaker_match and speaker_match.group(1) == target_speaker:
                            start_ms = time_to_ms(start_time)
                            end_ms = time_to_ms(end_time)
                            
                            start_ms = max(0, start_ms - padding_ms)  # Ensure it doesn't go below 0
                            end_ms = end_ms + padding_ms

                            if end_ms > start_ms:
                                segments.append((start_ms, end_ms))
        i += 1

    segments.sort(key=lambda x: x[0])
    

    if segments:
        merged_segments = [segments[0]]
        for start_ms, end_ms in segments[1:]:
            last_start, last_end = merged_segments[-1]
            if start_ms <= last_end:

                merged_segments[-1] = (last_start, max(last_end, end_ms))
            else:

                merged_segments.append((start_ms, end_ms))
        
        return merged_segments
    
    return segments

def extract_speaker_audio(wav_file, segments, output_file):
    """Extract audio segments for the specified speaker and save to a new file"""
    audio = AudioSegment.from_wav(wav_file)
    

    extracted_audio = AudioSegment.empty()
    for start_ms, end_ms in segments:
        segment = audio[start_ms:end_ms]
        extracted_audio += segment
    
    extracted_audio.export(output_file, format="wav")
    print(f"Extracted audio saved to {output_file}")
    print(f"Total duration: {len(extracted_audio)/1000:.2f} seconds")

def process_file_pair(srt_file, wav_file, output_dir, target_speaker=None):
    """Process a single SRT/WAV file pair, extracting the target speaker or all speakers"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(srt_file).rsplit('.', 1)[0]

    if target_speaker:
        output_file = os.path.join(output_dir, f"{base_name}_speaker{target_speaker}.wav")
        print(f"\nExtracting segments for Speaker {target_speaker} from {base_name}...")
        segments = parse_srt(srt_file, target_speaker)
        
        if segments:
            print(f"Found {len(segments)} segments for Speaker {target_speaker}")
            extract_speaker_audio(wav_file, segments, output_file)
        else:
            print(f"No segments found for Speaker {target_speaker}")
        
        return
    
    speakers = get_speakers_from_srt(srt_file)
    print(f"Found {len(speakers)} speaker(s) in {srt_file}: {', '.join(speakers)}")
    print("No speaker ID specified for this file. Use --speaker-id or --file-speakers to select a speaker.")

def process_directory(input_dir, output_dir, target_speaker=None, file_speakers=None):
    """Process all SRT and WAV files in the input directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    srt_files = [f for f in os.listdir(input_dir) if f.endswith('.srt')]
    
    if not srt_files:
        print(f"No SRT files found in {input_dir}")
        return
    
    print(f"Found {len(srt_files)} SRT file(s)")
    
    if target_speaker:
        print(f"Will extract only Speaker {target_speaker} from all files (unless overridden)")
    
    if file_speakers:
        print(f"Using specific speaker IDs for {len(file_speakers)} file(s)")
    

    for srt_file in srt_files:
        base_name = os.path.splitext(srt_file)[0]
        

        wav_file = os.path.join(input_dir, base_name + '.wav')
        if not os.path.exists(wav_file):
            print(f"Warning: No matching WAV file found for {srt_file}, skipping...")
            continue
        
        # Check if there's a specific speaker ID for this file
        file_speaker = None
        if file_speakers:
            if base_name in file_speakers:
                file_speaker = file_speakers[base_name]
            elif srt_file in file_speakers:
                file_speaker = file_speakers[srt_file]
            else:
                file_speaker = target_speaker
        else:
            file_speaker = target_speaker
        
        print(f"\nProcessing {srt_file} with {os.path.basename(wav_file)}...")
        if file_speaker:
            print(f"Targeting Speaker ID: {file_speaker}")
            
        process_file_pair(
            os.path.join(input_dir, srt_file),
            wav_file,
            output_dir,
            file_speaker
        )

def main():
    parser = argparse.ArgumentParser(description='Extract audio for speakers from WAV files using SRT timestamps.')
    parser.add_argument('--input-dir', '-i', required=True, help='Directory containing SRT and WAV files')
    parser.add_argument('--output-dir', '-o', default='separated_speakers', help='Output directory for extracted audio files')
    parser.add_argument('--speaker-id', help='Extract only this speaker ID (default for all files)')
    parser.add_argument('--file-speakers', nargs='+', help='Specify speaker IDs for specific files in format "filename:speakerid"')
    parser.add_argument('--single-file', '-s', action='store_true', help='Process a single file pair instead of a directory')
    parser.add_argument('--srt-file', help='Path to a single SRT file (use with --single-file)')
    parser.add_argument('--wav-file', help='Path to a single WAV file (use with --single-file)')
    
    args = parser.parse_args()
    
    # Parse file-speaker mappings
    file_speakers = {}
    if args.file_speakers:
        for mapping in args.file_speakers:
            if ':' in mapping:
                filename, speaker = mapping.split(':', 1)
                file_speakers[filename] = speaker
            else:
                print(f"Warning: Ignoring invalid file-speaker mapping: {mapping}")
    
    if args.single_file:
        if not args.srt_file or not args.wav_file:
            parser.error("--single-file requires both --srt-file and --wav-file")
        

        target_speaker = args.speaker_id
        if file_speakers:
            srt_basename = os.path.basename(args.srt_file)
            base_name = os.path.splitext(srt_basename)[0]
            
            if base_name in file_speakers:
                target_speaker = file_speakers[base_name]
                print(f"Using specific speaker ID {target_speaker} for {srt_basename}")
            elif srt_basename in file_speakers:
                target_speaker = file_speakers[srt_basename]
                print(f"Using specific speaker ID {target_speaker} for {srt_basename}")
        
        process_file_pair(args.srt_file, args.wav_file, args.output_dir, target_speaker)
    else:
        process_directory(args.input_dir, args.output_dir, args.speaker_id, file_speakers)

if __name__ == "__main__":
    main()
