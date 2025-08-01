import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback


def normalize_audio(input_file, output_file, target_loudness=-23.0):
    audio, sample_rate = sf.read(input_file)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize loudness
    meter = pyln.Meter(sample_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, target_loudness)

    # Peak normalize to -1 dB
    max_amplitude = np.max(np.abs(loudness_normalized_audio))
    peak_normalized_audio = loudness_normalized_audio / max_amplitude * 0.9  # -1 dB is approximately 0.9

    sf.write(output_file, peak_normalized_audio, sample_rate)


def process_file(input_file, output_file):
    try:
        normalize_audio(input_file, output_file)
        return f"Processed: {input_file}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error processing {input_file}: {str(e)}\n{tb}"


def process_directory(input_dir, output_dir, num_workers=1):  # Set to 1 for minimal memory usage
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for filename in input_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            futures.append(executor.submit(process_file, input_path, output_path))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing audio"):
            print(future.result())


if __name__ == "__main__":
    input_dir = "../Automatic-Audio-Dataset-Maker/WAVS1"
    output_dir = "../Automatic-Audio-Dataset-Maker/WAVS"
    
    print("Starting audio normalization process...")
    process_directory(input_dir, output_dir, num_workers=1)  # Try 1 or 2
    print("Audio normalization complete!")