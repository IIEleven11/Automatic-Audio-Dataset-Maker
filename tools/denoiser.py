import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from datasets import load_dataset
from tqdm import tqdm
import io
from tools.constants import COMBINED_USERNAME_REPOID, WAVS_DIR_POSTDENOISE



def denoise_audio(data, rate):
    # Convert to float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.6, stationary=True)
    
    return reduced_noise

def process_dataset(dataset_name, split, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Process each audio sample
    for index, sample in enumerate(tqdm(dataset, desc="Denoising audio")):
        audio = sample['audio']
        audio_data = audio['array']
        sampling_rate = audio['sampling_rate']
        
        # Denoise the audio
        denoised_audio = denoise_audio(audio_data, sampling_rate)
        
        # Denoise the audio
        denoised_audio = denoise_audio(audio_data, sampling_rate)
        
        # Save the denoised audio
        output_path = os.path.join(output_folder, f"denoised_audio_{index}.wav")
        sf.write(output_path, denoised_audio, sampling_rate)

if __name__ == "__main__":
    split = "train"
    print("Starting denoising process...")
    process_dataset(COMBINED_USERNAME_REPOID, split, WAVS_DIR_POSTDENOISE)
    print("Denoising complete!")