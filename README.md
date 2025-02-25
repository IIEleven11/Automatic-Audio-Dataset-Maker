# Audio Dataset Maker
Package conflict with windows machines and pesq/brouhaha. Suggest using WSL instead.
Curating datasets is extremely time consuming and tedious. I needed a way to automate this process as much as possible. 

**__Goal__**: Automate the creation and curation of an audio dataset for fine-tuneing/training text-to-speech models.

## What this project does: ##
   1. Transcription using whisper/~~deepgram. I am using their API because it is significantly faster. Using whipser is now an option.~~ (With the release of whisper turbo the deepgram implementation is now obsolete.)
   2. Segmentation and forcing a gaussian distribution of text/audio data segments between 2-18 seconds long.
   3. Creation of metadata according to the segmented audio. This is the transcriptions to pair with the audio files.
   4. Analyzing the audio using the SI-SDR, PESQ, STOI, c50, and SNR metrics.
   5. Filters the dataset, removing any audio that does not meet the threshold according to those metrics
   6. Creates a Huggingface Hub dataset repository as well as places the dataset on your drive

> Automation requires a high degree of reliability and consistency to be effective. Unfortunately, current speaker diarization technology does not meet the rigorous standards necessary for fully automated transcription. For optimal results, it is recommended to use this project with a single speaker dataset.

> While the script does include some speaker diarization options, their use is strongly discouraged. Attempting speaker diarization with the current technology will likely create more work than it saves due to the need for manual correction and verification.

## Installation

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

