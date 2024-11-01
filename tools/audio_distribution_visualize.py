# Plot and visualize the distribution of audio lengths in a directory of WAV files.
# We're checking the segmentation process. Script will produce an image for you.


import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_audio_lengths(directory):
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    durations = []
    
    print("Analyzing audio files...")
    for wav_file in tqdm(wav_files):
        try:
            duration = librosa.get_duration(path=os.path.join(directory, wav_file))
            durations.append(duration)
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")

    durations = np.array(durations)

    stats = {
        'count': len(durations),
        'mean': np.mean(durations),
        'median': np.median(durations),
        'std': np.std(durations),
        'min': np.min(durations),
        'max': np.max(durations)
    }
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(durations, bins=50, edgecolor='black')
    plt.title('Distribution of Audio Lengths')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')

    stats_text = (f"Total files: {stats['count']}\n"
                 f"Mean: {stats['mean']:.2f}s\n"
                 f"Median: {stats['median']:.2f}s\n"
                 f"Std Dev: {stats['std']:.2f}s\n"
                 f"Min: {stats['min']:.2f}s\n"
                 f"Max: {stats['max']:.2f}s")
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig('audio_length_distribution.png')
    plt.close()

    return stats

if __name__ == "__main__":
    directory = "WAVS_DIR_PREDENOISE" # Change this to the directory containing your WAV files
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist!")
    else:
        stats = analyze_audio_lengths(directory)
        
        print("\nAudio Length Statistics:")
        print(f"Total number of files: {stats['count']}")
        print(f"Mean duration: {stats['mean']:.2f} seconds")
        print(f"Median duration: {stats['median']:.2f} seconds")
        print(f"Standard deviation: {stats['std']:.2f} seconds")
        print(f"Minimum duration: {stats['min']:.2f} seconds")
        print(f"Maximum duration: {stats['max']:.2f} seconds")
        print("\nHistogram has been saved as 'audio_length_distribution.png'")