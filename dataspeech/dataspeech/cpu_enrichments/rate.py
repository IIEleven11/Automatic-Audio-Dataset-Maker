from g2p import make_g2p

transducer = make_g2p('eng', 'eng-ipa')

def clean_text(text):
    """Clean and validate text for g2p processing"""
    if text is None:
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove any non-ASCII characters that might cause issues
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove any escape sequences that might cause regex issues
    text = text.replace('\\', '')

    # Ensure there's at least some content
    if not text.strip():
        return "empty"

    return text

def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):
        speaking_rates = []
        phonemes_list = []
        if "speech_duration" in batch:
            for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
                try:
                    cleaned_text = clean_text(text)
                    phonemes = transducer(cleaned_text).output_string
                    audio_duration = audio_duration if audio_duration != 0 else 0.01
                    speaking_rate = len(phonemes) / audio_duration
                    speaking_rates.append(speaking_rate)
                    phonemes_list.append(phonemes)
                except Exception as e:
                    # Fallback for any errors
                    speaking_rates.append(0.0)
                    phonemes_list.append("")
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                try:
                    cleaned_text = clean_text(text)
                    phonemes = transducer(cleaned_text).output_string

                    sample_rate = audio["sampling_rate"]
                    audio_length = len(audio["array"].squeeze()) / sample_rate

                    speaking_rate = len(phonemes) / audio_length

                    speaking_rates.append(speaking_rate)
                    phonemes_list.append(phonemes)
                except Exception as e:
                    # Fallback for any errors
                    speaking_rates.append(0.0)
                    phonemes_list.append("")

        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        try:
            cleaned_text = clean_text(batch[text_column_name])
            phonemes = transducer(cleaned_text).output_string

            if "speech_duration" in batch:
                audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
            else:
                sample_rate = batch[audio_column_name]["sampling_rate"]
                audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate

            speaking_rate = len(phonemes) / audio_length

            batch["speaking_rate"] = speaking_rate
            batch["phonemes"] = phonemes
        except Exception as e:
            # Fallback for any errors
            batch["speaking_rate"] = 0.0
            batch["phonemes"] = ""

    return batch