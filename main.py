
import asyncio
from Step1_Transcribe_Audio import transcribe_audio_main

def main():
    audio_dir_path = input("Enter the path to the directory containing audio files: ")
    output_dir = input("Enter the path to the directory for saving transcriptions: ")

    asyncio.run(transcribe_audio_main(audio_dir_path, output_dir))

if __name__ == "__main__":
    main()
