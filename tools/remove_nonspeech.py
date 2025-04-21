import os
import wave
import contextlib
import webrtcvad
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import array
from tqdm import tqdm
import argparse

def read_wave(path):
    """Reads a .wav file, returns the PCM data and sample rate."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate, num_channels, sample_width

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def process_audio(input_path, output_path, aggressiveness=2):
    """Process audio file to remove non-speech segments."""
    # Read the audio file
    audio = AudioSegment.from_wav(input_path)
    
    # Convert to mono and 16kHz if necessary
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    
    # Convert to PCM data
    pcm_data = array.array('h', audio.raw_data)
    pcm_data = np.array(pcm_data, dtype=np.int16).tobytes()
    
    # Initialize VAD
    vad = webrtcvad.Vad(aggressiveness)
    
    # Process in frames
    frame_duration_ms = 30
    frames = frame_generator(frame_duration_ms, pcm_data, 16000)
    speech_frames = []
    
    # Detect speech frames
    for frame in frames:
        if len(frame) < 960:  # Skip frames that are too short
            continue
        is_speech = vad.is_speech(frame, 16000)
        if is_speech:
            speech_frames.append(frame)
    
    # Combine speech frames
    if speech_frames:
        speech_audio = b''.join(speech_frames)
        
        # Convert back to AudioSegment
        speech_segment = AudioSegment(
            data=speech_audio,
            sample_width=2,
            frame_rate=16000,
            channels=1
        )
        
        # Export the result
        speech_segment.export(output_path, format="wav")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Remove non-speech segments from WAV files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing WAV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed files')
    parser.add_argument('--aggressiveness', type=int, choices=[0, 1, 2, 3], default=2,
                      help='VAD aggressiveness (0-3, 3 being the most aggressive)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all wav files
    wav_files = list(Path(args.input_dir).glob('*.wav'))
    
    print(f"\nProcessing Configuration:")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"VAD Aggressiveness: {args.aggressiveness}")
    print(f"Number of files to process: {len(wav_files)}\n")
    
    successful = 0
    failed = 0
    
    for wav_path in tqdm(wav_files, desc="Processing audio files"):
        output_path = os.path.join(args.output_dir, wav_path.name)
        
        try:
            if process_audio(str(wav_path), output_path, args.aggressiveness):
                successful += 1
            else:
                failed += 1
                print(f"\nNo speech detected in {wav_path.name}")
        except Exception as e:
            failed += 1
            print(f"\nError processing {wav_path.name}: {str(e)}")
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")

if __name__ == "__main__":
    main()