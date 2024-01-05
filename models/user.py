"""Model simulating a human user, including adversarial inputs."""

import json
import numpy as np

class User:
    def __init__(self, mode='dataset', data_path=None, model_id=None):
        """Simulation of user submitting queries to worker model.

        Args:
            mode (str): Either 'dataset' if using canned queries or 'model'
                if simulating user with a model
            data_path (str): Path to canned queries
            model_id (str): Path to Hugging Face base model
        """
        
        self.mode = mode

        if self.mode == 'dataset':
            if data_path is None:
                raise ValueError("`data_path` must be set to use 'dataset' mode.")
            self.generator = iter(self.stream_jsonl(data_path))            
        elif self.mode == 'model':
            raise ValueError("'model' mode not yet supported. Please use 'dataset' mode.")
        else:
            raise ValueError("`mode` must be either 'dataset' or 'model'.")

    def stream_jsonl(self, data_path):
        """Create iterable generator from dataset in dataset mode.

        Args:
            data_path (str): Path to canned questions
        """
        
        with open(data_path, 'r') as file:
            lines = file.readlines()
        np.random.shuffle(lines)
        
        for line in lines:
            yield json.loads(line)

    def get_query(self):
        """Get a query from the user."""

        if self.mode == 'dataset':
            return next(self.generator)['chosen'].split('\n')[2]
        elif self.mode == 'model':
            raise ValueError("'model' mode not yet supported. Please use 'dataset' mode.")
        else:
            raise ValueError("mode must be either 'dataset' or 'model'.")
