# This is a very sensitive script. It can also be extremely heavy to run if you're denoising larger files. If it fails
# try changing the chink_size = rate * 30 to something like 15

import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from tqdm import tqdm
import gc

def denoise_audio(data, rate):
    try:
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # If big file make chunky
        chunk_size = rate * 20  # 30 seconds of audio
        if len(data) > chunk_size:
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                if len(chunk) == 0 or np.all(chunk == 0):
                    continue
                
                chunk = chunk.astype(np.float32)
                denoised_chunk = nr.reduce_noise(y=chunk, sr=rate, prop_decrease=0.6, stationary=True)
                chunks.append(denoised_chunk)
                
                # Garbageman
                gc.collect()
            
            return np.concatenate(chunks)
        else:
            # Process normally if file is small
            if len(data) == 0 or np.all(data == 0):
                return data
            
            data = data.astype(np.float32)
            return nr.reduce_noise(y=data, sr=rate, prop_decrease=0.6, stationary=True)
            
    except ValueError as e:
        print(f"Warning: Error during denoising - {str(e)}. Returning original audio.")
        return data
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return data

def process_dataset(dataset_name, split, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the dataset
    dataset = dataset(dataset_name, split=split)
    
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
    

'''
def process_local_wav_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    for wav_file in tqdm(wav_files, desc="Denoising audio files"):
        try:
            input_path = os.path.join(input_folder, wav_file)
            output_path = os.path.join(output_folder, f"denoised_{wav_file}")
            audio_data, sampling_rate = sf.read(input_path)
            denoised_audio = denoise_audio(audio_data, sampling_rate)
            sf.write(output_path, denoised_audio, sampling_rate)
            
            # Garbageman
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
            continue '''

if __name__ == "__main__":
    input_folder = "/home/eleven/AudioDatasetMaker/RAW_AUDIO/temp"
    output_folder = "/home/eleven/AudioDatasetMaker/WAVS_DIR_POSTDENOISE"
    
    print("Starting denoising process...")
    process_dataset(input_folder, output_folder)
    print("Denoising complete!")
