import os
import argparse
import soundfile as sf
import librosa
import tqdm
from pathlib import Path

def convert_audio(input_dir, output_dir=None, sample_rate=24000, bit_depth=16, mono=True):
    """
    Convert all audio files in a directory to WAV format with specified parameters.
    
    Args:
        input_dir (str): Directory containing audio files to convert
        output_dir (str, optional): Directory to save converted files. If None, uses input_dir/..WAV/
        sample_rate (int, optional): Target sample rate in Hz. Default is 24000.
        bit_depth (int, optional): Target bit depth. Default is 16.
        mono (bool, optional): Whether to convert to mono. Default is True.
    """
    input_dir = Path(input_dir)
    

    if output_dir is None:
        output_dir = input_dir.parent / 'WAVS'
    else:
        output_dir = Path(output_dir)
    

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting audio files from {input_dir} to {output_dir}")
    print(f"Parameters: Sample rate={sample_rate}Hz, Bit depth={bit_depth}, Mono={mono}")
    
    audio_files = []
    for ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']:
        audio_files.extend(list(input_dir.glob(f'*{ext}')))
        audio_files.extend(list(input_dir.glob(f'*{ext.upper()}')))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to convert")

    for audio_file in tqdm.tqdm(audio_files):
        try:
            output_file = output_dir / f"{audio_file.stem}.wav"
            
            y, sr = librosa.load(audio_file, sr=sample_rate, mono=mono)
            
            subtype = 'PCM_16' if bit_depth == 16 else 'PCM_24' if bit_depth == 24 else 'FLOAT'
            sf.write(output_file, y, sample_rate, subtype=subtype)
            
        except Exception as e:
            print(f"Error converting {audio_file}: {e}")
    
    print(f"Conversion complete. Converted {len(audio_files)} files to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files to WAV with specified parameters")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing audio files to convert")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save converted files")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate in Hz")
    parser.add_argument("--bit_depth", type=int, default=16, help="Target bit depth (16 or 24)")
    parser.add_argument("--mono", action="store_true", default=True, help="Convert to mono")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    if input_dir is None:
        # Get the default RAW_AUDIO directory from the project structure
        script_dir = Path(__file__).resolve().parent
        project_dir = script_dir.parent
        input_dir = project_dir / "RAW_AUDIO"
    
    convert_audio(
        input_dir=input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        mono=args.mono
    )