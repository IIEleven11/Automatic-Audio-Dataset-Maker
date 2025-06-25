import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
import argparse

def de_echo_audio(audio_data, sr, method='spectral_subtraction'):
    """
    Apply de-echoing/dereverberation to audio data
    """
    if method == 'spectral_subtraction':
        # Apply spectral subtraction for noise/echo reduction
        stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor (assuming first 0.5 seconds is representative)
        noise_frames = int(0.5 * sr / 512)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        clean_magnitude = magnitude - alpha * noise_spectrum
        clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=512)
        
    elif method == 'wiener_filter':
        # Simple Wiener filtering approach
        stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Estimate signal and noise power
        signal_power = magnitude ** 2
        noise_power = np.mean(signal_power[:, :int(0.5 * sr / 512)], axis=1, keepdims=True)
        
        # Wiener filter
        wiener_gain = signal_power / (signal_power + noise_power)
        clean_stft = stft * wiener_gain
        clean_audio = librosa.istft(clean_stft, hop_length=512)
        
    else:
        # High-pass filter to remove low-frequency echo components
        nyquist = sr / 2
        low_cutoff = 80 / nyquist
        b, a = signal.butter(4, low_cutoff, btype='high')
        clean_audio = signal.filtfilt(b, a, audio_data)
    
    return clean_audio

def process_wav_files(input_folder, output_folder=None, method='spectral_subtraction'):
    """
    Process all WAV files in the input folder
    """
    input_path = Path(input_folder)
    
    if output_folder is None:
        output_folder = input_path / "de_echoed"
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    wav_files = list(input_path.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {input_folder}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    print(f"Output folder: {output_path}")
    
    for i, wav_file in enumerate(wav_files):
        try:
            print(f"Processing ({i+1}/{len(wav_files)}): {wav_file.name}")
            
            # Load audio file
            audio_data, sr = librosa.load(wav_file, sr=None)
            
            # Apply de-echoing
            clean_audio = de_echo_audio(audio_data, sr, method=method)
            
            # Normalize to prevent clipping
            clean_audio = clean_audio / np.max(np.abs(clean_audio)) * 0.95
            
            # Save processed audio
            output_file = output_path / f"de_echoed_{wav_file.name}"
            sf.write(output_file, clean_audio, sr)
            
            print(f"  ✓ Saved: {output_file}")
            
        except Exception as e:
            print(f"  ✗ Error processing {wav_file.name}: {str(e)}")
    
    print(f"\nProcessing complete! Processed files saved in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='De-echo WAV files in a folder')
    parser.add_argument('--input', '-i', 
                       default='/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/DEREVERBED',
                       help='Input folder containing WAV files')
    parser.add_argument('--output', '-o', 
                       help='Output folder (default: input_folder/de_echoed)')
    parser.add_argument('--method', '-m', 
                       choices=['spectral_subtraction', 'wiener_filter', 'highpass'],
                       default='spectral_subtraction',
                       help='De-echoing method to use')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        return
    
    process_wav_files(args.input, args.output, args.method)

if __name__ == "__main__":
    main()