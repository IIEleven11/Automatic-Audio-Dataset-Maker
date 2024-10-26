import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from datetime import timedelta
from accelerate import InitProcessGroupKwargs


logger = get_logger(__name__, log_level="INFO")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_name_or_path: str = field(
        metadata={"help": "The name of the model to use (via the transformers library) for the prompt annotation."},
    )
    per_device_eval_batch_size: int = field(
        metadata={"help": "The per-device batch size to use for inference."},
    )
    model_variant: str = field(
        default=None,
        metadata={"help": "If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. "},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and the computations run. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={"help": "Which attn type to use: ['eager', 'sdpa', 'flash_attention_2']"},
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 8-bit precision for inference."}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 4-bit precision for inference."}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "Use fast tokenizer for encoding/decoding input ids"}
    )
    token: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to use an authentication token when loading/uploading from the Hugging Face Hub"
        },
    )
    do_sample: Optional[bool] = field(default=True, metadata={"help": "Whether to use sampling mode for generation"})
    temperature: Optional[float] = field(default=0.6, metadata={"help": "Temperature for sampling-based generation"})
    max_new_tokens: Optional[int] = field(
        default=256, metadata={"help": "Maximum number of new tokens during generation"}
    )
    torch_compile: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to compile the forward pass (not sampling) in generate. Only compatible with Gemma and LlaMA."
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        metadata={
            "help": "Where to save the processed dataset to disk. If unspecified, uses a 'pretty' version of the "
            "original dataset name. E.g. 'facebook/voxpopuli' will be saved under 'voxpopuli'."
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples for generation - use for debugging purposes."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of processes to use for the dataloader."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    hub_dataset_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository namespace if pushing to the Hugging Face Hub."},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory each time the script is run."},
    )
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Save the generated prompts every save_steps."},
    )
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": ("If a value is passed, will limit the total number of saved checkpoints")}
    )
    speaker_name: Optional[str] = field(
        default=None,
        metadata={"help": "If `is_single_speaker`, it specified the speaker name that you want to give to the mono-speaker of your dataset."},
    )
    is_single_speaker: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a single speaker prompt, with a single name, specified by `speaker_name`."}
    )
    is_new_speaker_prompt: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use the newest speaker prompt, which will be used for the next Parler-TTS."}
    )
    speaker_id_column: Optional[str] = field(
        default=None, metadata={"help": "Speaker id column name. Only used if creating a dataset with multiple speaker names (i.e if `speaker_ids_to_name_json` is specified)"}
    )
    speaker_ids_to_name_json: Optional[str] = field(
        default=None, metadata={"help": "Path to a JSON file which map some speaker ids to some names. Only used if `speaker_id_column` is specified."}
    )
    accent_column: Optional[str] = field(
        default=None, metadata={"help": "Accent column name, if any."}
    )


    def __post_init__(self):
        if self.push_to_hub and self.hub_dataset_id is None:
            raise ValueError("You must specify the `hub_dataset_id` when setting `--push_to_hub=True`")


def get_quantization_config(model_args: ModelArguments) -> Union[BitsAndBytesConfig, None]:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Union[Dict[str, int], None]:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


CHECKPOINT_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+).json$")


def save_checkpoint(output_dir, all_generated_ids, step):
    checkpoint_path = f"{CHECKPOINT_PREFIX}-{step}.json"
    output_path = os.path.join(output_dir, checkpoint_path)
    all_generated_ids = [ids.tolist() for ids in all_generated_ids]
    with open(output_path, "w") as file:
        json.dump(all_generated_ids, file)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "r") as file:
        all_generated_ids = json.load(file)
    logger.info(f"Json file {checkpoint_path} loaded.")
    all_generated_ids = [np.array(lst) for lst in all_generated_ids]
    return all_generated_ids


def sorted_checkpoints(output_dir=None) -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{CHECKPOINT_PREFIX}-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{CHECKPOINT_PREFIX}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        os.remove(checkpoint)


def get_last_checkpoint(folder, return_list=False) -> Tuple[List, int]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return [], 0
    content = os.listdir(folder)
    checkpoints = [path for path in content if _RE_CHECKPOINT.search(path) is not None]
    if len(checkpoints) == 0:
        return [], 0
    last_checkpoint = os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))
    # Find num steps saved state string pattern
    pattern = r"checkpoint-(\d+).json"
    match = re.search(pattern, last_checkpoint)
    cur_step = int(match.group(1))
    if return_list:
        # load corresponding generated ids
        all_generated_ids = load_checkpoint(last_checkpoint)
        return all_generated_ids, cur_step
    else:
        return [], cur_step


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch.
    """

    tokenizer: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_ids = {"input_ids": [feature["input_ids"] for feature in features]}
        batch = self.tokenizer.pad(input_ids, return_tensors="pt", padding="longest", return_attention_mask=True)
        return batch


PROMPT = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (e.g., male, female)
2. The level of reverberation (e.g., very roomy sounding, quite roomy sounding, slightly roomy sounding, moderate reverberation, slightly confined sounding, quite confined sounding, very confined sounding)
3. The amount of noise the sample (e.g., very noisy, quite noisy, slightly noisy, moderate ambient sound, slightly clear, quite clear, very clear)
4. The tone of the speaker's voice (e.g., very monotone, quite monotone, slightly monotone, moderate intonation, slightly expressive, quite expressive, very expressive)
5. The pace of the speaker's delivery (e.g., very slowly, quite slowly, slightly slowly, moderate speed, slightly fast, quite fast, very fast)
6. The pitch of the speaker's voice (e.g., very low pitch, quite low pitch, slightly low pitch, moderate pitch, slightly high pitch, quite high pitch, very high pitch)
Your task is to create a text description using these keywords that accurately describes the speech sample while ensuring the description remains grammatically correct and easy to understand. You should rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. If the amount of noise is 'very noisy' and the level of reverberation is 'very roomy sounding', include terms like 'very bad recording' in the description. Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very confined sounding', include terms like 'very good recording' in the description. Otherwise, do not add extra details beyond what has been provided, and only return the generated description.
For example, given the following keywords: 'female', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'slightly low pitch', 'very slowly', a valid description would be: 'a woman with a deep voice speaks slowly but has an animated delivery in an echoey room with some background noise'.
For the keywords: '[gender]', '[reverberation]', '[noise]', '[speech_monotony]', '[pitch]', '[speaking_rate]', the corresponding description is:"
"""

NEW_PROMPT = """You will be given six descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (male, female)
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)

Your task is to create a text description using these keywords that accurately describes the speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'female', 'slightly distant-sounding', 'noisy', 'very expressive and animated', 'very slowly', 'moderate pitch', a valid description would be: 'A woman speaks very slowly but has a very animated delivery. The recording is noisy and there is some roominess.'
Another valid description would be: 'In a noisy room, a female speaker delivers a very animated and expressive speech, at a very slow pace.'
Another valid description would be: 'A woman enunciates a very expressive speech. Her voice is slightly distant-sounding, with some background noise present. She speaks very slowly with a moderate pitch but a very expressive tone.'

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', the corresponding description is:
"""

NEW_PROMPT_WITH_ACCENT = """You will be given 7 descriptive keywords related to an audio sample of a person's speech. These keywords include:
1. The gender (male, female)
2. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
4. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
5. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)
6. The pitch of the speaker's voice (very low-pitch, low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch)
7. The accent of the speaker.

Your task is to create a text description using these keywords that accurately describes the speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'female', 'slightly distant-sounding', 'noisy', 'very expressive and animated', 'very slowly', 'moderate pitch', 'Chinese', a valid description would be: 'A woman with a Chinese accent speaks very slowly but has a very animated delivery. The recording is noisy and there is some roominess.'
Another valid description would be: 'In a noisy room, a female speaker with a Chinese accent delivers a very animated and expressive speech, at a very slow pace.'
Another valid description would be: 'A woman with a Chinese accent enunciates a very expressive speech. Her voice is slightly distant-sounding, with some background noise present. She speaks very slowly with a moderate pitch but a very expressive tone.'

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: '[gender]', '[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', '[pitch]', '[accent]', the corresponding description is:
"""


NEW_SINGLE_SPEAKER_PROMPT = """You will be given four descriptive keywords related to an audio sample of [speaker_name]'s speech. These keywords include:
1. The level of reverberation (very distant-sounding, distant-sounding, slightly distant-sounding, slightly close-sounding, very close-sounding)
3. The amount of noise in the sample (extremely noisy, very noisy, noisy, slightly noisy, almost no noise, very clear)
3. The tone of the speaker's voice (very monotone, monotone, slightly expressive and animated, expressive and animated, very expressive and animated)
4. The pace of the speaker's delivery (very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast)

Your task is to create a text description using these keywords that accurately describes [speaker_name]'s speech sample.
If the amount of noise is 'very noisy' and the level of reverberation is 'very distant-sounding', you must include terms such as 'very poor recording' or `very bad recording` in the description. 
Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very close-sounding', you must include terms like 'very good recording' or `excellent recording` in the description. 
You can randomly omit the following terms, as they are default terms: 'moderate speed' and 'moderate pitch'.
Do not add extra details beyond what has been provided above. You can change the order of keywords, and replace synonymous terms.

For example, given the following keywords: 'slightly distant-sounding', 'clear', 'very expressive and animated', 'slightly fast', a valid description would be: '[speaker_name] speaks slightly fast but has a very animated delivery in a room with slight echo but no background noise.'
Another valid description would be: `In a very animated voice, [speaker_name] delivers words slightly quickly. The room is quite, but there's a bit of echo.'

Ensure that the generated description is grammatically correct, easy to understand, and concise. Only return one and only one description.

For the keywords: ''[reverberation]', '[sdr_noise]', '[speech_monotony]', '[speaking_rate]', the corresponding description is:
"""

SINGLE_SPEAKER_PROMPT = """You will be given four descriptive keywords related to an audio sample of [speaker_name]'s speech. These keywords include:
1. The level of reverberation (e.g., very roomy sounding, quite roomy sounding, slightly roomy sounding, moderate reverberation, slightly confined sounding, quite confined sounding, very confined sounding)
2. The amount of noise the sample (e.g., very noisy, quite noisy, slightly noisy, moderate ambient sound, slightly clear, quite clear, very clear)
3. The tone of the speaker's voice (e.g., very monotone, quite monotone, slightly monotone, moderate intonation, slightly expressive, quite expressive, very expressive)
4. The pace of the speaker's delivery (e.g., very slowly, quite slowly, slightly slowly, moderate speed, slightly fast, quite fast, very fast)

Your task is to create a single and only short text description using these keywords that accurately describes the speech sample while ensuring the description remains grammatically correct and easy to understand. You should rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. If the amount of noise is 'very noisy' and the level of reverberation is 'very roomy sounding', you must include terms like 'very bad recording' in the description. Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very confined sounding', you must include terms like 'very good recording' in the description. Otherwise, do not add extra details beyond what has been provided, and only return the generated description.

For example, given the following keywords: 'slightly roomy sounding', 'quite noisy', 'very expressive', 'very slowly', a valid description would be: '[speaker_name] speaks very slowly but has an animated delivery in an echoey room with background noise.'.
Feel free to change the order of keywords, and to use synonyms, for example, with the previous keywords: `In a very expressive voice, [speaker_name] pronounces her words incredibly slowly. There's some background noise in this room with a bit of echo.'.

For the keywords: ''[reverberation]', '[noise]', '[speech_monotony]', '[speaking_rate]', the corresponding description is:
"""

def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if data_args.is_single_speaker and data_args.speaker_name is None:
        raise ValueError("`is_single_speaker=True` but `speaker_name` is not specified. Specify it or remove `is_single_speaker`.")

    if not data_args.is_single_speaker and data_args.speaker_name:
        raise ValueError(f"`is_single_speaker=False` but `speaker_name=data_args.speaker_name` is not specified. Add `--is_single_speaker` or remove `speaker_name`.")


    # Create the custom configuration
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600*3))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    if data_args.overwrite_output_dir and os.path.exists(data_args.output_dir) and os.path.isdir(data_args.output_dir):
        logger.info("Cleaning output dir from previous run...")
        shutil.rmtree(data_args.output_dir)

    # 3. Load annotated dataset
    logger.info("*** Load annotated dataset ***")
    if data_args.dataset_split_name is not None:
        raw_datasets = DatasetDict()
        data_splits = data_args.dataset_split_name.split("+")
        # load on a split-wise basis
        for split in data_splits:
            with accelerator.local_main_process_first():
                raw_datasets[split] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    num_proc=data_args.preprocessing_num_workers,
                )
    else:
        with accelerator.local_main_process_first():
            # load all splits for annotation
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )

    raw_datasets_features = set(raw_datasets[next(iter(raw_datasets))].features.keys())

    if data_args.max_eval_samples is not None:
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    EXPECTED_COLUMNS = {"gender", "pitch", "noise", "reverberation", "speech_monotony", "speaking_rate"}
    if data_args.is_single_speaker:
        EXPECTED_COLUMNS = {"noise", "reverberation", "speech_monotony", "speaking_rate"}
        
    if data_args.is_new_speaker_prompt:
        EXPECTED_COLUMNS.remove("noise")
        EXPECTED_COLUMNS.add("sdr_noise")
        
    speaker_ids_to_name = {}
    speaker_id_column = data_args.speaker_id_column
    if data_args.speaker_id_column and data_args.speaker_ids_to_name_json:
        import json
        if data_args.is_single_speaker:
            raise ValueError(f"`is_single_speaker=True` but `speaker_ids_to_name_json={data_args.speaker_ids_to_name_json}`. Specify one or another.")
        
        EXPECTED_COLUMNS.add(data_args.speaker_id_column)
        with open(data_args.speaker_ids_to_name_json, "r") as read_file:
            speaker_ids_to_name = json.load(read_file)

    if not EXPECTED_COLUMNS.issubset(raw_datasets_features):
        missing_columns = EXPECTED_COLUMNS - raw_datasets_features
        raise ValueError(
            f"Missing columns {missing_columns} from the dataset features. Got dataset features {raw_datasets_features}"
        )

    # 4. Load pre-trained model
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        variant=model_args.model_variant,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        token=model_args.token,
    ).eval()

    if model_args.torch_compile:
        # torch compile only compatible with gemma and llama
        if not callable(getattr(model, "_setup_cache", None)):
            raise ValueError(
                f"Static k/v cache is not compatible with the model {model.__class__.__name__}. Set `--torch_compile=False"
                "for dynamic k/v cache"
            )
        model.generation_config.cache_implementation = "static"
        # compile the forward pass (but not the top-{p,k} sampling)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    speaker_name = data_args.speaker_name
    is_single_speaker = data_args.is_single_speaker
    is_new_speaker_prompt = data_args.is_new_speaker_prompt
    accent_column_name = data_args.accent_column

    def prepare_dataset(sample):
        sample_prompt = PROMPT
        if is_single_speaker:
            sample_prompt = SINGLE_SPEAKER_PROMPT if not is_new_speaker_prompt else NEW_SINGLE_SPEAKER_PROMPT
            sample_prompt = sample_prompt.replace(f"[speaker_name]", speaker_name)
        elif (speaker_id_column and speaker_ids_to_name.get(str(sample.get(speaker_id_column)), None)):
            name =  speaker_ids_to_name.get(str(sample.get(speaker_id_column)), None)
            sample_prompt = SINGLE_SPEAKER_PROMPT if not is_new_speaker_prompt else NEW_SINGLE_SPEAKER_PROMPT
            sample_prompt = sample_prompt.replace(f"[speaker_name]", name)
        elif is_new_speaker_prompt and accent_column_name is not None:
            sample_prompt = NEW_PROMPT if sample.get(accent_column_name, "Unindentified") == "Unindentified" else NEW_PROMPT_WITH_ACCENT
        elif is_new_speaker_prompt:
            sample_prompt = NEW_PROMPT
        for key in EXPECTED_COLUMNS:
            sample_prompt = sample_prompt.replace(f"[{key}]", sample[key])
        if accent_column_name is not None and sample.get(accent_column_name, "Unindentified") != "Unindentified":
            sample_prompt = sample_prompt.replace("[accent]", sample["accent"])
            
        sample_prompt = [{"role": "user", "content": sample_prompt}]
        token_ids = tokenizer.apply_chat_template(sample_prompt)
        sample["input_ids"] = token_ids
        return sample

    with accelerator.local_main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset, num_proc=data_args.preprocessing_num_workers, desc="Preparing prompts"
        )

    # Prepare everything with our `accelerator`
    model = accelerator.prepare(model)
    data_collator = DataCollatorWithPadding(tokenizer)

    def generate_step(batch):
        output_ids = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=model_args.do_sample,
            temperature=model_args.temperature,
            max_new_tokens=model_args.max_new_tokens,
        )
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    def postprocess_dataset(batch):
        prompt_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        generated_texts = tokenizer.batch_decode(batch["generated_ids"], skip_special_tokens=True)
        
        batch["text_description"] = [generated_text[len(prompt_text) :] for (prompt_text, generated_text) in zip(prompt_texts, generated_texts)]
        return batch

    for split in vectorized_datasets:
        data_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=model_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=data_args.dataloader_num_workers,
            pin_memory=True,
        )
        data_loader = accelerator.prepare(data_loader)
        total_inference_steps = len(data_loader)
        progress_bar = tqdm(
            range(total_inference_steps), desc=" ... ", position=0, disable=not accelerator.is_local_main_process
        )

        split_output_dir = os.path.join(data_args.output_dir, split)
        all_generated_ids, cur_step = get_last_checkpoint(split_output_dir, accelerator.is_local_main_process)
        accelerator.wait_for_everyone()

        if cur_step > 0:
            logger.info(f"Resuming {split} from step {cur_step}")
            # efficiently skip the first n batches
            data_loader = skip_first_batches(data_loader, cur_step)
            progress_bar.update(cur_step)

        while cur_step < total_inference_steps:
            for batch in data_loader:
                generated_ids = generate_step(batch)
                generated_ids = accelerator.gather_for_metrics(generated_ids)
                if accelerator.is_local_main_process:
                    all_generated_ids.extend(generated_ids.cpu().numpy())

                cur_step += 1
                progress_bar.update(1)

                if (cur_step % data_args.save_steps == 0) or (cur_step == total_inference_steps):
                    if accelerator.is_main_process:
                        save_checkpoint(split_output_dir, all_generated_ids, cur_step)
                        rotate_checkpoints(data_args.save_total_limit, output_dir=split_output_dir)
                    accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            vectorized_datasets[split] = vectorized_datasets[split].add_column("generated_ids", all_generated_ids)

        if accelerator.is_main_process:
            vectorized_datasets[split] = vectorized_datasets[split].map(
                postprocess_dataset,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Postprocessing dataset",
                remove_columns=["input_ids", "generated_ids"],
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        vectorized_datasets.save_to_disk(data_args.output_dir)
        if data_args.push_to_hub:
            vectorized_datasets.push_to_hub(
                data_args.hub_dataset_id,
                config_name=data_args.dataset_config_name if data_args.dataset_config_name is not None else "default",
                token=model_args.token,
            )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()