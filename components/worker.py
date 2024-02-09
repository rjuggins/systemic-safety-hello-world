"""Class for Worker AI model, i.e. the deployed AI actually trying to complete tasks."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import cuda
from accelerate import infer_auto_device_map, init_empty_weights
from langchain import PromptTemplate, FewShotPromptTemplate
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

        # Create worker prompt template
        self.create_prompt_template()

    def create_prompt_template(self, examples=[]):
        """Create FewShotPromptTemplate with hard-coded worker structure.
        
        Args:
            examples (list(dict)): Dictionaries of 'query', 'answer' pairs
                containing few shot examples
        """

        example_template = """
        Human: {query}
        AI: {answer}
        """

        example_prompt = PromptTemplate(
            input_variables = ["query", "answer"],
            template=example_template
        )

        prefix = """
        You are an AI responding to questions from a human. Try and be helpful but not harmful.
        Some examples of good behaviour are:
        """

        suffix = """
        {query}
        AI:
        """

        self.prompt_template = FewShotPromptTemplate(
            examples=[],
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["query"],
            example_separator="\n"
        )

    def set_prompt_examples(self, examples):
        """Set examples in FewShotPromptTemplate.
        
        Args:
            examples (list(dict)): Dictionaries of 'query', 'answer' pairs
                containing few shot examples
        """

        self.prompt_template.examples = examples

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
            use_auth_token=self.hf_auth,
            padding_side='left'
        )
        self.tokenizer.pad_token=self.tokenizer.eos_token

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
        queries,
        min_additional_len=50,
        num_return_sequences=1,
        temperature=1.0
        ):
        """Generate text from a prompt.

        Args:
            queries (list[str]): Queries for worker model
            min_additional_len (int): Number of tokens in response for longest query
            num_return_sequences (int): Number of different responses per query
            temperature (float): Sampling temperature

        Returns:
            list[str]: Responses to queries
        """

        # Add queries to prompt template
        prompts = [self.prompt_template.format(query=query) for query in queries]

        # Set max length based on maximum prompt length and desired min additional length
        prompt_tokens = [self.tokenizer.tokenize(prompt) for prompt in prompts]
        max_prompt_len = max([len(prompt) for prompt in prompt_tokens])
        max_length = max_prompt_len + min_additional_len
        
        # Tokenize input text
        input_ids = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)['input_ids']

        # Generate text
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        responses = [self.tokenizer.decode(response, skip_special_tokens=True) for response in output]

        return responses