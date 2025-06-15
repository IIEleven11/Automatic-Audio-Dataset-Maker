import soundfile as sf
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import librosa
import argparse

def convert_to_mono(input_path, output_path, target_sr=None):
    """Convert stereo audio file to mono and optionally resample."""
    try:
        audio, original_sr = librosa.load(input_path, sr=None)  # Load without resampling first
        
        if len(audio.shape) == 2 and audio.shape[1] == 2:
            mono_audio = np.mean(audio, axis=1)
        else:
            mono_audio = audio
            

        if target_sr and target_sr != original_sr:
            print(f"Resampling {input_path} from {original_sr}Hz to {target_sr}Hz")
            mono_audio = librosa.resample(
                y=mono_audio,
                orig_sr=original_sr,
                target_sr=target_sr
            )
            final_sr = target_sr
        else:
            final_sr = original_sr
            

        sf.write(output_path, mono_audio, final_sr)
        

        verify_audio, verify_sr = librosa.load(output_path, sr=None)
        if verify_sr != final_sr:
            raise Exception(f"Sample rate verification failed. Expected {final_sr}Hz, got {verify_sr}Hz")
            
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():

    parser = argparse.ArgumentParser(description='Convert audio files to mono and optionally change sample rate.')
    parser.add_argument('--sample_rate', type=int, choices=[22050, 24000, 44100, 48000],
                      help='Target sample rate (optional)')
    parser.add_argument('--input_dir', type=str, 
                      default="/home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/RAW_AUDIO1",
                      help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str,
                      default="/home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/RAW_AUDIO",
                      help='Output directory for processed files')
    args = parser.parse_args()
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    wav_files = list(Path(args.input_dir).glob('*.wav'))
    

    successful = 0
    failed = 0
    
 
    print("\nProcessing Configuration:")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Sample Rate: {args.sample_rate if args.sample_rate else 'Original (unchanged)'}")
    print(f"Number of files to process: {len(wav_files)}\n")
    
    for wav_path in tqdm(wav_files, desc="Converting audio files"):
        output_path = os.path.join(args.output_dir, wav_path.name)
        
        # Get original sample rate
        audio, orig_sr = librosa.load(str(wav_path), sr=None)
        print(f"\nProcessing {wav_path.name} (Original SR: {orig_sr}Hz)")
        
        if convert_to_mono(str(wav_path), output_path, args.sample_rate):
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion complete:")
    print(f"Successfully converted: {successful} files")
    print(f"Failed to convert: {failed} files")

if __name__ == "__main__":
    main()