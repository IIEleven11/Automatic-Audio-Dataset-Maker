# AudioDatasetMaker
An attempt to create an audio dataset ready for fine tuning a voice model without needing to listen to the audio.
Uses deepgram/whisper/custom models to create an LJSpeech dataset for voice model fine tuning.

## Installation

1. conda create -n audiodatasetmaker python=3.10
2. conda activate audiodatasetmaker
3. pip install -r requirements.txt
4. Install git-lifs
   - Linux (Ubuntu): sudo apt-get install git-lfs 
   - Windows: https://git-lfs.com/ download then:  git lfs install 
5. Get a deepgram API key from https://deepgram.com/dashboard/signup
6. Set HUGGINGFACE_TOKEN environment variable within your OS.

## Usage
1. Put your audio files in the RAW_AUDIO folder
2. Run python main.py and follow the prompts in the terminal to insert speaker name, eval percentage, and API key


### In progress:
- The next portion of the project is will analyse the audio and filter it using several different metrics.