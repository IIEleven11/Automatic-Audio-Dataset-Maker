import subprocess
import os




local_dir ="./Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model"

def main():
    download_cmd = "huggingface-cli download KimberleyJSN/melbandroformer --local-dir /workspace/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model"
    subprocess.run(download_cmd, shell=True, check=True)
    
    if not os.path.exists("DENOISED"):
        os.makedirs("DENOISED")

    inference_cmd = "python /workspace/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/inference.py --config_path /workspace/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/configs/config_vocals_mel_band_roformer.yaml --model_path /home/eleven/__REPOS__/cormier_female_1/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/models/mel_band_roformer/MelBandRoformer.ckpt --input_folder /workspace/Automatic-Audio-Dataset-Maker/RAW_AUDIO --store_dir /workspace/Automatic-Audio-Dataset-Maker/RAW_AUDIO1"
    subprocess.run(inference_cmd, shell=True, check=True)

if __name__ == "__main__":
    main()

