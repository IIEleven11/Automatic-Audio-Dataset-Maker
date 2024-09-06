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
  - Note: There are several different metrics you can adjust within the scripts. Or you can just use what I have set as default. I chose values that I found to be the most successful or logical. That being said, all 
    datasets are different.If you are not getting a result you like, experiment.
  - This will create a gaussian distribution of audio segments ranging from 1.2 to 15 seconds long with a max character length of 250. It will create a metadata_train.csv, metadata_eval.csv, a wavs folder full of the 
    segmented audio, a raw JSON/SRT file and a metadata.csv file
  - Pipe delimited with the header being audio_file|text|speaker_name. This can be easily adjusted if youre using a different format.

### In progress:
- The next portion of the project will analyse the audio and filter it using several different metrics.
