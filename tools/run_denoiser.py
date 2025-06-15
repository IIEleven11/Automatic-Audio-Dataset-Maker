import subprocess
import os

def main():
<<<<<<< HEAD
    download_cmd = "huggingface-cli download KimberleyJSN/melbandroformer --local-dir models/mel_band_roformer"
    subprocess.run(download_cmd, shell=True, check=True)
    

    if not os.path.exists("DENOISED"):
        os.makedirs("DENOISED")

    inference_cmd = "python /home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/inference.py --config_path /home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/configs/config_vocals_mel_band_roformer.yaml --model_path /home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/models/mel_band_roformer/MelBandRoformer.ckpt --input_folder /home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/RAW_AUDIO --store_dir /home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/RAW_AUDIO1"
    subprocess.run(inference_cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
=======
    # Download the model file
    download_cmd = "huggingface-cli download KimberleyJSN/melbandroformer --local-dir ./Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/models/mel_band_roformer"
    subprocess.run(download_cmd, shell=True, check=True)
    
    # Rename downloaded file to expected name
    #if os.path.exists("models/mel_band_roformer"):
     #   os.rename("models/mel_band_roformer", "melbandroformer.ckpt")
    
    # Create output directory if it doesn't exist
    if not os.path.exists("DENOISED"):
        os.makedirs("DENOISED")
    
    # Run the inference
    inference_cmd = "python ./Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/inference.py --config_path ./Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/configs/config_vocals_mel_band_roformer.yaml --model_path ./Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/models/mel_band_roformer/MelBandRoformer.ckpt --input_folder ./Automatic-Audio-Dataset-Maker/RAW_AUDIO --store_dir ./Automatic-Audio-Dataset-Maker/DENOISED"
    subprocess.run(inference_cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
>>>>>>> 56f0673ad2084b8d03c5402f657231a67a2b75f3
