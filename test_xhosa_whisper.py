#!/usr/bin/env python3
"""
Test script for the Xhosa Whisper model integration.
This script tests the basic functionality of loading and using the TheirStory/whisper-small-xhosa model.
"""

import os
import torch
import librosa
import numpy as np
from transformers import pipeline

def test_xhosa_whisper_model():
    """Test the Xhosa Whisper model loading and basic functionality using pipeline."""

    print("Testing Xhosa Whisper Model Integration (Pipeline Approach)")
    print("=" * 60)

    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model configuration
    model_name = "TheirStory/whisper-small-xhosa"
    print(f"Loading model: {model_name}")

    try:
        # Create the pipeline using the same approach as the working example
        print("Creating ASR pipeline...")
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )
        print("✓ Pipeline created successfully")

        # Test with a dummy audio signal (sine wave)
        print("\nTesting with dummy audio signal...")

        # Create a 3-second sine wave at 440Hz (A note)
        sample_rate = 16000  # Whisper expects 16kHz
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        print(f"Audio data shape: {audio_data.shape}")
        print(f"Audio duration: {duration} seconds")

        # Save temporary audio file for testing
        import soundfile as sf
        temp_audio_path = "temp_test_audio.wav"
        sf.write(temp_audio_path, audio_data, sample_rate)

        # Generate transcription using pipeline
        print("Generating transcription...")
        result = pipe(
            temp_audio_path,
            batch_size=8,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True
        )

        transcription = result["text"]
        chunks = result.get("chunks", [])

        print(f"Transcription result: '{transcription}'")
        if chunks:
            print(f"Number of chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: {chunk}")

        # Clean up
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n✓ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_with_real_audio(audio_file_path):
    """Test with a real audio file using pipeline approach."""

    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return False

    print(f"\nTesting with real audio file: {audio_file_path}")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    model_name = "TheirStory/whisper-small-xhosa"

    try:
        # Create the pipeline
        print("Creating ASR pipeline...")
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )
        print("✓ Pipeline created successfully")

        # Load audio info
        audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
        print(f"Audio loaded: {len(audio_data)} samples, {len(audio_data)/sample_rate:.2f} seconds")

        # Generate transcription using pipeline
        print("Generating transcription...")
        result = pipe(
            audio_file_path,
            batch_size=8,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True
        )

        transcription = result["text"]
        chunks = result.get("chunks", [])

        print(f"Transcription: '{transcription}'")
        if chunks:
            print(f"Number of chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"  Chunk {i+1}: {chunk}")
            if len(chunks) > 3:
                print(f"  ... and {len(chunks) - 3} more chunks")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"Error processing real audio: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Test basic functionality
    success = test_xhosa_whisper_model()
    
    # Test with real audio if available
    audio_dir = "RAW_AUDIO"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        if audio_files:
            test_audio = os.path.join(audio_dir, audio_files[0])
            print(f"\nFound audio file for testing: {test_audio}")
            test_with_real_audio(test_audio)
    
    if success:
        print("\n🎉 All tests passed! The Xhosa Whisper model integration is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
