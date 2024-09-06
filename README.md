# AudioDatasetMaker
An attempt to create an audio dataset ready for fine tuning a voice model without needing to listen to the audio.
Uses deepgram/whisper/custom models to create an LJSpeech dataset for voice model fine tuning.

## Installation

1. conda create -n audiodatasetmaker python=3.10
2. conda activate audiodatasetmaker
3. pip install -r requirements.txt
4. Get deepgram API key

## Usage
1. Put your audio files in the RAW_AUDIO folder
2. Run python main.py and follow the prompts in the terminal to insert speaker name, eval percentage, and API key


### In progress:
- The next portion of the project is will analyse the audio and filter it using several different metrics.