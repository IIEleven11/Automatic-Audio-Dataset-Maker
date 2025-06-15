#!/usr/bin/env python3
"""
Script to convert audio files in RAW_AUDIO to WAV format (24000Hz, 16-bit, mono)
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to import the tools module
script_path = Path(__file__).resolve()
project_root = script_path.parent
sys.path.append(str(project_root))

# Import the convertaudio module from the tools package
from tools.convertaudio import convert_audio

if __name__ == "__main__":
    # Define the input directory (RAW_AUDIO)
    input_dir = project_root / "RAW_AUDIO"
    
    # Define the output directory (WAVS)
    output_dir = project_root / "WAVS"
    
    # Run the conversion with specified parameters
    convert_audio(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_rate=24000,
        bit_depth=16,
        mono=True
    )