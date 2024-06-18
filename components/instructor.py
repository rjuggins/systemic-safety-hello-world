"""Class to instruction-tune base models and get sufficient performance to run system.
This is distinct from the Teacher class, which is used to update models during the running
of the system."""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import pandas as pd
from datasets import Dataset
from torch import cuda
from functools import partial
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


class Instructor:
    def __init__(self, config):
        """Class that instruction-tunes base models.

        Args:
            config (dict): Configuration parameters
                Must contain:
                    model_id (str): Path to Hugging Face base model
                    data_path (str): Path to instruction-tuning dataset
                    instruction_schema (dict): Keys for user and assistant content in dataset,
                        plus context if exists (which is appended to user content)
                May contain:
                    hf_auth (str): Authorisation token for Hugging Face, e.g. for Llama 2
                    max_length (int): Max sequence length for model
        """

        self.model_id = config["model_id"]
        self.data_path = config["instruction_data_path"]
        self.schema = config["instruction_schema"]
        self.quantization = config["quantization"]

        hf_key_path = config.get("hf_key_path")
        if hf_key_path is not None:
            with open(hf_key_path, "r") as file:
                self.hf_auth = file.read().strip()
        else:
            self.hf_auth = None

        # If max_length defined in config, use that value, if not use default
        if config.get("max_length") is None:
            self.max_length = 512
        else:
            self.max_length = config["max_length"]

        # If sliding_window defined in config, use that value, if not use default
        if config.get("sliding_window") is None:
            self.sliding_window = self.max_length // 2
        else:
            self.sliding_window = config["sliding_window"]

        self.device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        model_config = AutoConfig.from_pretrained(
            self.model_id, token=self.hf_auth
        )
        
        self.device_map = 'auto'

    def load_model(self):
        """Load model weights and initialise tokenizer."""

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=self.hf_auth, padding_side="right" # left is for inference
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.quantization == True:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=self.bnb_config,
                device_map=self.device_map,
                token=self.hf_auth
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                token=self.hf_auth
            )

        self.model.config.window = self.sliding_window
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        print(f"Model loaded on {self.device}")

    def load_data(self, test_size=0.0):
        """Load instruction dataset, reformat, and load into Dataset objects.

        Args:
            test_size (float): Fraction of data to use for validation
        """

        # Load raw data
        instruction_data = []
        with open(self.data_path, "r") as file:
            for line in file:
                instruction_data.append(json.loads(line))

        # Create keys for components of instruction data
        user = self.schema["user"]
        context = self.schema["context"]
        assistant = self.schema["assistant"]

        # Reformat data with role/content structure
        examples = []
        for instruction in instruction_data:
            instruction_message = [{"role": "user"}, {"role": "assistant"}]

            if len(instruction[context]) > 0:
                instruction_message[0]["content"] = (
                    instruction[user] + " " + instruction[context]
                )
            else:
                instruction_message[0]["content"] = instruction[user]
            instruction_message[1]["content"] = instruction[assistant]
            examples.append(instruction_message)

        # Apply chat template to examples
        examples = [
            self.tokenizer.apply_chat_template(instruction, tokenize=False)
            for instruction in examples
        ]

        # Load into Dataset classes
        if test_size == 0:
            self.train_dataset = Dataset.from_dict({'text':examples})
            self.test_dataset = None
        else:
            full_dataset = Dataset.from_dict({'text':examples})

            # Split the dataset into training and testing
            split_datasets = full_dataset.train_test_split(test_size=test_size)
            self.train_dataset = split_datasets["train"]
            self.test_dataset = split_datasets["test"]

    @staticmethod
    def encode(examples, tokenizer, max_length):
        """Static method to map tokenizer over dataset. Fails if tries to use self.tokenizer."""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
            )

    def tokenize_data(self):
        """Tokenize train and test datasets."""

        # Prefill self.encode tokenizer argument
        encode_with_tokenizer = partial(
            self.encode, tokenizer=self.tokenizer, max_length=self.max_length
            )

        # Tokenize train dataset
        self.train_dataset = self.train_dataset.map(encode_with_tokenizer, batched=True)
        self.train_dataset = self.train_dataset.remove_columns("text")

        # If exists, tokenize test dataset
        if self.test_dataset is not None:
            self.test_dataset = self.test_dataset.map(encode_with_tokenizer, batched=True)
            self.test_dataset = self.test_dataset.remove_columns("text")

    def tune_model(self):
        """Instruction-tune model."""

        pass
