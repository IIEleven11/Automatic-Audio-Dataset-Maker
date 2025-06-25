import subprocess
import os

input_folder = "/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/RAW_AUDIO"
model_path = "/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/models/MelBandRoformer.ckpt"
config_path = "/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/configs/config_vocals_mel_band_roformer.yaml"
store_dir = "/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/denoised"


def main():
    if not os.path.exists("models/mel_band_roformer"):
        os.makedirs("models/mel_band_roformer")
        print("Created directory: models/mel_band_roformer")
        
    if not os.path.exists("/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/models/MelBandRoformer.ckpt"):
        print("Downloading model...")
        subprocess.run("wget https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt -P /home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/models/MelBandRoformer.ckpt", shell=True, check=True)
        print("Downloaded model")
    

    if not os.path.exists("DENOISED"):
        os.makedirs("DENOISED")

    inference_cmd = '''python /home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/tools/denoiser/Mel-Band-Roformer-Vocal-Model/inference.py --input_folder {input_folder} --model_path {model_path} --config_path {config_path} --store_dir {store_dir}'''.format(input_folder=input_folder, model_path=model_path, config_path=config_path, store_dir=store_dir)
    subprocess.run(inference_cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
