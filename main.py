from Step1_Transcribe_Audio import transcribe_audio
from Step2_Convert_JSON2SRT import convert_json_to_srt
from Step3_Segment_Create_Metadata import segment_and_create_metadata
from Step4_split_metadata import split_metadata

def main():
    transcribe_audio()
    convert_json_to_srt()
    segment_and_create_metadata()
    split_metadata()

if __name__ == "__main__":
    main()
