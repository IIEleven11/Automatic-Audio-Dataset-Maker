import os
from dotenv import load_dotenv
from deepgram import Deepgram
import json


load_dotenv()

# Get the Deepgram API key from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

dg_client = Deepgram(DEEPGRAM_API_KEY)

# Directory containing the audio files you want to transcribe
AUDIO_DIR_PATH = "/home/eleven/makeADataset/raw_Audio" # change to raw audio path

JSON_DIR_PATH = "/home/eleven/deepgram/jsons" #change to jsons path


os.makedirs(JSON_DIR_PATH, exist_ok=True)

# Transcription options, edit as needed
options = {
    "model": "whisper-large", # adjust this if you want to use a different model that deepgram offers. (Whisper-large is more accurate but slower)
    #"language": "es, en, ", change this if you need a different language. See: https://developers.deepgram.com/docs/deepgram-whisper-cloud#supported-languages for language codes
    "punctuate": True,
    "utterances": True,
    "paragraphs": True,
    "smart_format": True,
    "filler_words": True
}

async def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            audio_source = {"buffer": audio_file, "mimetype": "audio/wav"}

            response = await dg_client.transcription.prerecorded(audio_source, options)
            print(f"Transcription response for {file_path}:", json.dumps(response, indent=2))


            base_name = os.path.splitext(os.path.basename(file_path))[0]
            json_file_name = f"{base_name}.json"
            json_file_path = os.path.join(JSON_DIR_PATH, json_file_name)


            with open(json_file_path, "w") as json_file:
                json.dump(response, json_file, indent=2)

            print(f"Transcription saved to {json_file_path}")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

async def main():
    audio_files = [os.path.join(AUDIO_DIR_PATH, f) for f in os.listdir(AUDIO_DIR_PATH) if f.endswith('.wav')]

    for audio_file in audio_files:
        await transcribe_audio(audio_file)


import asyncio
asyncio.run(main())
