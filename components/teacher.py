"""Class orchestrating training of worker. Includes running benchmarks and calling PMs.
This is distinct from the Instructor class, which is used to instruction-tune models before
they are loaded into the Worker class."""

from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import TrainingArguments
from trl import DPOTrainer
from huggingface_hub import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError


class Teacher:
    def __init__(self, config):
        """Class that uses DPO to teach a Worker that has been failed by an overseer.

        Args:
            config (dict): Configuration parameters
                Must contain:
                    lora_params (dict): LoraConfig parameters
                    teaching_params (dict): TrainingArguments parameters
                    dpo_beta (float): beta parameter for DPOTrainer
                May contain:
                    model_repo_id (str): Name of HF repo to push trained model to
                    checkpoint_name (str): Name of model checkpoint to be pushed
                    max_length (int): Max sequence length for DPOTrainer
                    max_prompt_length (int): Max prompt length for DPOTrainer
        """

        self.lora_params = config["lora_params"]
        self.training_params = config["teaching_params"]
        self.beta = config["dpo_beta"]
        self.model_repo_id = config.get("model_repo_id")
        self.checkpoint_name = config.get("checkpoint_name")

        # If max_length defined in config, use that value, if not use default
        if config.get("max_length") is None:
            self.max_length = 512
        else:
            self.max_length = config["max_length"]

        # If max_prompt_length defined in config, use that value, if not use default
        if config.get("max_prompt_length") is None:
            self.max_prompt_length = self.max_length // 2
        else:
            self.max_prompt_length = config["max_prompt_length"]

    @staticmethod
    def dpo_format(example):
        """Format Anthropic HH data as a dictionary with 'prompt', 'chosen', and 'rejected' keys."""

        chosen_list = example['chosen'].split()
        rejected_list = example['rejected'].split()
        response_index = len(chosen_list) - 1 - chosen_list[::-1].index('Assistant:')

        prompt = ' '.join(chosen_list[:response_index])
        chosen = ' '.join(chosen_list[response_index:])
        rejected = ' '.join(rejected_list[response_index:])

        return {
            'prompt':prompt,
            'chosen':chosen,
            'rejected':rejected
        }

    def load_data(self, data_path):
        """Load DPO datase into Dataset object.
        
        Args:
            data_path (str): Path to dpo dataset
        """

        dataset = load_dataset('json', data_files={'train': data_path})
        self.teaching_dataset = dataset['train'].map(self.dpo_format)

    # @staticmethod
    # def encode(examples, tokenizer, max_length):
    #     """Static method to map tokenizer over dataset. Fails if tries to use self.tokenizer."""
    #     return tokenizer(
    #         examples["text"],
    #         padding="max_length",
    #         truncation=True,
    #         max_length=max_length
    #         )

    # def tokenize_data(self):
    #     """Tokenize train and test datasets."""

    #     # Prefill self.encode tokenizer argument
    #     encode_with_tokenizer = partial(
    #         self.encode, tokenizer=self.tokenizer, max_length=self.max_length
    #         )

    #     # Tokenize train dataset
    #     self.train_dataset = self.train_dataset.map(encode_with_tokenizer, batched=True)
    #     self.train_dataset = self.train_dataset.remove_columns("text")

    #     # If exists, tokenize test dataset
    #     if self.test_dataset is not None:
    #         self.test_dataset = self.test_dataset.map(encode_with_tokenizer, batched=True)
    #         self.test_dataset = self.test_dataset.remove_columns("text")

    def train_model(self, worker):
        """Teach Worker model.

        Args:
            worker (Worker): Worker that has been failed by an overseer
        """

        # LoRA configuration
        peft_config = LoraConfig(**self.lora_params)

        new_worker = get_peft_model(worker.model, peft_config)
        new_worker.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(**self.training_params)

        # Create DPO trainer
        trainer = DPOTrainer(
            new_worker,
            ref_model=None,
            args=training_args,
            train_dataset=self.teaching_dataset,
            tokenizer=worker.tokenizer,
            peft_config=peft_config,
            beta=self.beta,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length
        )

        trainer.train()

    # def push_model(self):
    #     """Push model or checkpoint to Hugging Face Hub."""

    #     # Initialize API
    #     api = HfApi()

    #     # Function to check if the repo exists
    #     def repo_exists(repo_id, token):
    #         try:
    #             api.repo_info(repo_id=repo_id, token=token)
    #             return True
    #         except RepositoryNotFoundError:
    #             return False

    #     # Check if the repository exists
    #     if repo_exists(self.model_repo_id, self.hf_auth):
    #         print(f"Not pushing model as repository {self.model_repo_id} already exists.")
    #     else:
    #         if self.checkpoint_name is not None:
    #             model_dir = os.path.join(self.training_params["output_dir"], self.checkpoint_name)
    #             # Clear GPU cache
    #             torch.cuda.empty_cache()
    #             checkpoint_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto')
    #             checkpoint_model.push_to_hub(self.model_repo_id, use_auth_token=self.hf_auth)
    #         else:
    #             self.model.push_to_hub(self.model_repo_id, use_auth_token=self.hf_auth)

