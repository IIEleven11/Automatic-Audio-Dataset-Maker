# Audio Dataset Maker
Curating datasets is extremely time consuming and tedious. I needed a way to automate this process as much as possible. 

# Automatic Audio Dataset Maker

Automatic Audio Dataset Maker is a tool designed to automate the creation and curation of high-quality audio datasets, primarily for training text-to-speech models.

## Key Features

### Audio Processing
- Transcribes audio using local whisper models
- Segments audio n seconds of chunks with a gaussian distribution
- Creates metadata/transcriptions paired with audio segments

### Quality Control
- Analyzes audio using multiple metrics:
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - PESQ (Perceptual Evaluation of Speech Quality)
  - STOI (Short-Time Objective Intelligibility)
  - C50 (Clarity Index)
  - SNR (Signal-to-Noise Ratio)
- Filters out audio that doesn't meet quality thresholds

### Dataset Management
- Creates/Saves dataset to the huggingface hub as well as a local copy.
- Integrates with DataSpeech library for additional audio annotations and natural language descriptions

## Installation
-NOTE: Theres a package conflict on windows machines with pesq/brouhaha. I suggest using WSL/Linux instead.
1. conda create -n audiodatasetmaker python==3.10
2. conda activate audiodatasetmaker
3. pip install -r requirements.txt
4. Install git-lifs
   - Linux (Ubuntu): sudo apt-get install git-lfs 
   - Windows: https://git-lfs.com/ download then:  git lfs install 
6. Set HUGGINGFACE_TOKEN environment variable within your OS.
8. In your terminal login to Hugging Face Hub by typing: ```huggingface-cli login```

## Usage
1. Put your audio files in the RAW_AUDIO folder. They should be 24000hz, mono, and 16bit PCM. (These are not absolute values. I am just setting something as a default for any beginners to follow)
2. Setup config.yaml with your options then python adm_main.py -or-
   - Run python adm_main.py and follow the prompts in the terminal
   
   Example to run without config.yaml:
   1. Enter your Hugging Face username: __```IIEleven11```__
   2. Enter the repository name: __```MyRepositoryName```__
   3. Do you want to skip Step 1/2 (Transcribe and convert audio)? (y/n): __```n```__
   4. Enter the SPEAKER_NAME: __```Steve```__
   5. Enter the EVAL_PERCENTAGE (percentage of data to move to evaluation set): __```10```__

   #### Note: 
      - Step 1 (transcription) using local whisperASR.
      - This process will filter out any data it deems as not suitable for training. The denoising and normalizing process is very sensitive and can very easily make the process fail. I suggest doing any denoising or editing of the audio before hand then choosing to skip this part in the script when prompted. Check the tools folder for some scripts I use to            do this.
      - You can choose to skip the transcription step if you have your own. Make sure you place a metadata.csv, metadata_train.csv, and metadata_eval.csv in the appropriate folders.
      - Analyzing and computing the audio metrics can be a bit GPU intensive. My RTX 3090 can handle a few hours of data without a problem. I could see less capable hardware failing during this step. Any large amounts of data will require at least a 4090.
      -  You will end up with .parquet file containing a **curated** dataset including audio data. This will be on the huggingface hub under your username/repository name. As well as saved locally in the FILTERED_PARQUET folder.
      -  There us a script in the tools folder ```convert_dataspeech.py``` you can input the correct paths and run it to automatically convert the parquet file/s into metadata and get the wavs in a folder.


## If you install the Data Wrangler Extension within VsCode you can view the final parquet and it will look something like this.
![image](https://github.com/user-attachments/assets/b8690113-4a25-4582-8868-95afc5b2a061)

## Use these metrics to view the dataset and then open up filter_parquet.py and adjust the thresholds to have more control of the filtering process.

#### Note: *There are many different dataset formats. Each model will be slightly different. You can easily view, manipulate, and convert the .parquet dataset within an editor like vscode. I use the DataWrangler extension but there are many other ways.*

### Functionality/Updates:
- [x] The project will now transcribe, convert, segment, and split the audio into metadata.
- [x] Then it will analyze the audio and filter it using several different metrics.
- [x] Finally it will push the dataset to the Hugging Face Hub as well as save a local copy.
- [ ] Create .parquet conversion scripts for the different dataset formats that voice models can be trained on. eg: XTTSv2, StyleTTS2, Parler-TTS, FishSpeech, etc.
   - [ ] XTTSv2
   - [ ] StyleTTS2
   - [ ] Parler-TTS
   - [ ] FishSpeech
   - [ ] Tortoise-TTS







#### Citation:
- https://github.com/huggingface/dataspeech

- lacombe-etal-2024-dataspeech,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Data-Speech},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  how published = {\url{https://github.com/ylacombe/dataspeech}}

- lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}

