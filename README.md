# AudioDatasetMaker

Uses deepgram/whisper/custom models to create an LJSpeech dataset for voice model fine tuning.

### High level overview of the process
##### This process will result in a metadata_train.csv, metadata_eval.csv and a folder of segmented wav files
1. Create a deepgram transcription JSON.
2. Convert the transcription JSON to an SRT file with timestamps.
3. Segmentation using the SRT file while attempting to achieve a gaussian distribution of audio length.
4. Creation of the metadata files for training and evaluation.

## Installation

1. conda create -n audiodatasetmaker python=3.10
2. conda activate audiodatasetmaker
3. pip install -r requirements.txt
4. Get deepgram API key
5. Set DEEPGRAM_API_KEY environment variable

## Usage

Within each script you will find further instruction. Use each script in order. 
1. Step1_DgramTrans2JSON.py
2. Step2_Convert_JSON2Wav.py
3. Step3_SegmentandCreate_Metadata.py 
4. Step4_make_train_and_eval.py
