import logging
import time
import numpy as np
import torch
import penn
# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065


logger = logging.getLogger(__name__)

def pitch_apply(batch, rank=None, audio_column_name="audio", output_column_name="utterance_pitch", penn_batch_size=4096):
    MIN_AUDIO_LENGTH = 4000
    start_time = time.time()
    
    logger.info(f"Starting pitch_apply function with batch size: {len(batch[audio_column_name]) if isinstance(batch[audio_column_name], list) else 1}")
    
    if isinstance(batch[audio_column_name], list):  
        logger.info("Processing batch list mode")
        utterance_pitch_mean = []
        utterance_pitch_std = []
        for idx, sample in enumerate(batch[audio_column_name]):
            sample_start = time.time()
            logger.info(f"Processing sample {idx+1}/{len(batch[audio_column_name])} - Length: {len(sample['array'])}")
            
            # Pad audio if needed
            audio_array = sample["array"]
            if len(audio_array) < MIN_AUDIO_LENGTH:
                logger.info(f"Padding sample {idx+1} from length {len(audio_array)} to {MIN_AUDIO_LENGTH}")
                audio_array = np.pad(audio_array, (0, MIN_AUDIO_LENGTH - len(audio_array)))
            
            try:
                logger.info(f"Starting penn.from_audio for sample {idx+1}")
                # Infer pitch and periodicity
                pitch, periodicity = penn.from_audio(
                    torch.tensor(audio_array[None, :]).float(),
                    sample["sampling_rate"],
                    hopsize=hopsize,
                    fmin=fmin,
                    fmax=fmax,
                    checkpoint=checkpoint,
                    batch_size=penn_batch_size,
                    center=center,
                    interp_unvoiced_at=interp_unvoiced_at,
                    gpu=(rank or 0)% torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
                    )
                
                logger.info(f"Successfully processed sample {idx+1}")
                utterance_pitch_mean.append(pitch.mean().cpu())
                utterance_pitch_std.append(pitch.std().cpu())
                
                sample_duration = time.time() - sample_start
                logger.info(f"Sample {idx+1} processed in {sample_duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing sample {idx+1}: {str(e)}")
                utterance_pitch_mean.append(torch.tensor(0.0))
                utterance_pitch_std.append(torch.tensor(0.0))
            
        batch[f"{output_column_name}_mean"] = utterance_pitch_mean 
        batch[f"{output_column_name}_std"] = utterance_pitch_std 
    else:
        logger.info("Processing single sample mode")
        sample = batch[audio_column_name]
        # Pad audio if needed
        audio_array = sample["array"]
        original_length = len(audio_array)
        logger.info(f"Single sample length: {original_length}")
        
        if len(audio_array) < MIN_AUDIO_LENGTH:
            logger.info(f"Padding single sample from length {len(audio_array)} to {MIN_AUDIO_LENGTH}")
            audio_array = np.pad(audio_array, (0, MIN_AUDIO_LENGTH - len(audio_array)))
            
        try:
            logger.info("Starting penn.from_audio for single sample")
            pitch, periodicity = penn.from_audio(
                    torch.tensor(audio_array[None, :]).float(),
                    sample["sampling_rate"],
                    hopsize=hopsize,
                    fmin=fmin,
                    fmax=fmax,
                    checkpoint=checkpoint,
                    batch_size=penn_batch_size,
                    center=center,
                    interp_unvoiced_at=interp_unvoiced_at,
                    gpu=(rank or 0)% torch.cuda.device_count() if torch.cuda.device_count() > 0 else rank
                    )        
            batch[f"{output_column_name}_mean"] = pitch.mean().cpu()
            batch[f"{output_column_name}_std"] = pitch.std().cpu()
            logger.info("Successfully processed single sample")
        except Exception as e:
            logger.error(f"Error processing single sample: {str(e)}")
            batch[f"{output_column_name}_mean"] = torch.tensor(0.0)
            batch[f"{output_column_name}_std"] = torch.tensor(0.0)

    total_duration = time.time() - start_time
    logger.info(f"pitch_apply function completed in {total_duration:.2f} seconds")
    return batch