import os
import torch
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dereverb_model():
    """
    Load a dereverb model. This example uses torchaudio's spectral subtraction,
    but you can replace this with other models like:
    - DeepFilterNet
    - Facebook Denoiser
    - Custom trained models
    """
    try:
        # Option 1: Using torchaudio (basic spectral processing)
        import torchaudio
        logger.info("Using torchaudio spectral processing for dereverberation")
        return "torchaudio"
        
    except ImportError:
        logger.warning("torchaudio not available, using basic numpy implementation")
        return "numpy"

def spectral_subtraction_dereverb(audio, sr, alpha=2.0, beta=0.01):
    """
    Basic spectral subtraction for dereverberation
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise spectrum (use first few frames)
    noise_frames = 10
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Spectral subtraction
    enhanced_magnitude = magnitude - alpha * noise_spectrum
    enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
    
    # Reconstruct signal
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
    
    return enhanced_audio

def wiener_filter_dereverb(audio, sr):
    """
    Wiener filtering approach for dereverberation
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate power spectral density
    psd = magnitude ** 2
    
    # Estimate noise PSD (from beginning and end of signal)
    noise_frames = min(20, psd.shape[1] // 4)
    noise_psd = np.mean(np.concatenate([
        psd[:, :noise_frames], 
        psd[:, -noise_frames:]
    ], axis=1), axis=1, keepdims=True)
    
    # Wiener filter
    wiener_gain = psd / (psd + noise_psd + 1e-10)
    
    # Apply filter
    enhanced_magnitude = magnitude * wiener_gain
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
    
    return enhanced_audio

def deep_filter_net_dereverb(audio, sr):
    """
    Use DeepFilterNet for dereverberation (if available)
    Install with: pip install deepfilternet
    """
    try:
        from df.enhance import enhance, init_df
        from df.io import resample
        
        # Initialize DeepFilterNet
        model, df_state, _ = init_df()
        
        # Resample if needed
        if sr != df_state.sr():
            audio_resampled = resample(audio, sr, df_state.sr())
        else:
            audio_resampled = audio
            
        # Enhance audio
        enhanced = enhance(model, df_state, audio_resampled.reshape(1, -1))
        
        # Resample back if needed
        if sr != df_state.sr():
            enhanced = resample(enhanced.squeeze(), df_state.sr(), sr)
        else:
            enhanced = enhanced.squeeze()
            
        return enhanced
        
    except ImportError:
        logger.warning("DeepFilterNet not available, falling back to spectral subtraction")
        return spectral_subtraction_dereverb(audio, sr)
    except Exception as e:
        logger.error(f"Error with DeepFilterNet: {e}, falling back to spectral subtraction")
        return spectral_subtraction_dereverb(audio, sr)

def facebook_denoiser_dereverb(audio, sr):
    """
    Use Facebook Denoiser for dereverberation (if available)
    Install with: pip install denoiser
    """
    try:
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
        
        # Load pretrained model
        model = pretrained.dns64()
        model.eval()
        
        # Convert audio to torch tensor
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Resample if needed (Facebook denoiser expects 16kHz)
        if sr != 16000:
            wav = convert_audio(wav, sr, 16000, 1)
            
        # Denoise
        with torch.no_grad():
            enhanced = model(wav)
            
        # Convert back to numpy and resample if needed
        enhanced = enhanced.squeeze().cpu().numpy()
        if sr != 16000:
            enhanced = librosa.resample(enhanced, orig_sr=16000, target_sr=sr)
            
        return enhanced
        
    except ImportError:
        logger.warning("Facebook Denoiser not available, falling back to spectral subtraction")
        return spectral_subtraction_dereverb(audio, sr)
    except Exception as e:
        logger.error(f"Error with Facebook Denoiser: {e}, falling back to spectral subtraction")
        return spectral_subtraction_dereverb(audio, sr)

def dereverb_audio_file(input_path, output_path, method="auto", target_sr=22050):
    """
    Dereverberate a single audio file
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None)
        
        # Choose dereverberation method
        if method == "auto":
            # Try methods in order of preference
            try:
                enhanced_audio = deep_filter_net_dereverb(audio, sr)
                used_method = "DeepFilterNet"
            except:
                try:
                    enhanced_audio = facebook_denoiser_dereverb(audio, sr)
                    used_method = "Facebook Denoiser"
                except:
                    enhanced_audio = wiener_filter_dereverb(audio, sr)
                    used_method = "Wiener Filter"
        elif method == "deepfilternet":
            enhanced_audio = deep_filter_net_dereverb(audio, sr)
            used_method = "DeepFilterNet"
        elif method == "facebook":
            enhanced_audio = facebook_denoiser_dereverb(audio, sr)
            used_method = "Facebook Denoiser"
        elif method == "wiener":
            enhanced_audio = wiener_filter_dereverb(audio, sr)
            used_method = "Wiener Filter"
        elif method == "spectral":
            enhanced_audio = spectral_subtraction_dereverb(audio, sr)
            used_method = "Spectral Subtraction"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Resample if needed
        if sr != target_sr:
            enhanced_audio = librosa.resample(enhanced_audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Normalize audio
        enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio) + 1e-8)
        
        # Save enhanced audio
        sf.write(output_path, enhanced_audio, sr)
        
        return True, used_method
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False, str(e)

def dereverb_folder(input_dir, output_dir, method="auto", target_sr=22050):
    """
    Dereverberate all audio files in a folder
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported audio extensions
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    success_count = 0
    method_counts = {}
    
    for audio_file in tqdm(audio_files, desc="Dereverbating audio files"):
        # Create output filename
        output_filename = audio_file.stem + ".wav"  # Always save as .wav
        output_file_path = output_path / output_filename
        
        # Skip if already processed
        if output_file_path.exists():
            logger.info(f"Skipping {audio_file.name} - already processed")
            continue
        
        # Process file
        success, used_method = dereverb_audio_file(
            str(audio_file), 
            str(output_file_path), 
            method=method,
            target_sr=target_sr
        )
        
        if success:
            success_count += 1
            method_counts[used_method] = method_counts.get(used_method, 0) + 1
            logger.info(f"✓ Successfully processed {audio_file.name} using {used_method}")
        else:
            logger.error(f"✗ Failed to process {audio_file.name}: {used_method}")
    
    # Print summary
    logger.info(f"\nDereverberation complete!")
    logger.info(f"Successfully processed: {success_count}/{len(audio_files)} files")
    logger.info("Methods used:")
    for method, count in method_counts.items():
        logger.info(f"  {method}: {count} files")

def main():
    """
    Main function to run the dereverberation script
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Dereverberate audio files")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/DENOISED",
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/eleven/__REPOS__/AADM_Clean/Automatic-Audio-Dataset-Maker/DEREVERBED",
        help="Output directory for dereverbed audio files"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["auto", "deepfilternet", "facebook", "wiener", "spectral"],
        default="auto",
        help="Dereverberation method to use"
    )
    parser.add_argument(
        "--sample_rate", 
        type=int, 
        default=22050,
        help="Target sample rate for output files"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting dereverberation process...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Target sample rate: {args.sample_rate}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Run dereverberation
    dereverb_folder(
        args.input_dir, 
        args.output_dir, 
        method=args.method,
        target_sr=args.sample_rate
    )

if __name__ == "__main__":
    main()