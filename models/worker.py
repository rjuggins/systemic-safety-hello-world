"""Class for Worker AI model, i.e. the deployed AI actually trying to complete tasks."""

import torch
from torch import cuda
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


class Worker:
    def __init__(self, model_id, mode='eval', hf_auth=None):
        """Chatbot responding to requests.

        Args:
            model_id (str): Path to Hugging Face base model
            mode (str): If 'eval' will set model.eval()
            hf_auth (str): Authorisation token for Hugging Face, e.g. for Llama 2
        """

        self.model_id = model_id
        self.mode = mode
        self.hf_auth = hf_auth

        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")

        model_config = AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(model_config)

    def check_device_map(self, no_split_module_classes=[]):
        """Check model fits on GPUs.
        Args:
            no_split_module_classes (list(str)): Class names of layers not to split
                between devices
        """

        self.device_map = infer_auto_device_map(
            self.model,
            no_split_module_classes=no_split_module_classes
        )
        print(self.device_map)

    def load_model(self):
        """Load model weights and initialise tokenizer."""

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            use_auth_token=self.hf_auth
        )

        if self.mode == 'eval':
            self.model.eval()

        print(f"Model loaded on {self.device}")

    def generate_text(
        self,
        prompt,
        max_response_len=50,
        num_return_sequences=1,
        temperature=1.0
        ):
        """Generate text from a prompt."""

        # Set max length based on prompt length and desired max response length
        prompt_tokens = self.tokenizer.tokenize(prompt)
        max_length = len(prompt_tokens) + max_response_len
        
        # Tokenize input text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate text
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response
