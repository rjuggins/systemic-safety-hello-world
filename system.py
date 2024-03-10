"""Class orchestrating AI system."""

import yaml
import numpy as np

from components.worker import Worker
from components.outside_expert import OutsideExpert
from components.user import User
from components.overseer import Overseer


class System:
    def __init__(self):
        """Initialising models in system."""

        # Load config file into dictionary
        with open('./config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialise Worker
        self.worker = Worker(self.config['model_id'])
        self.worker.check_device_map(no_split_module_classes=self.config['no_split_module_classes'])
        self.worker.load_model()
        with open('./config/worker_prompt_examples.yaml', 'r') as file:
            worker_prompt_examples = yaml.safe_load(file)
        self.worker.create_prompt_template(worker_prompt_examples)

        # Initialise User
        self.user = User(data_path=self.config['user_data_path'])

        # Initialise OutsideExperts
        with open(self.config['openai_key_path'], 'r') as file:
            openai_key = file.read().strip()
        with open('./config/expert_prompt_examples.yaml', 'r') as file:
            expert_prompt_examples = yaml.safe_load(file)

        self.helpfulness_expert = OutsideExpert(
            model_id=self.config['expert_model_id'],
            openai_key=openai_key,
            target='helpfulness',
            example=expert_prompt_examples['helpfulness']
            )

        self.harmlessness_expert = OutsideExpert(
            model_id=self.config['expert_model_id'],
            openai_key=openai_key,
            target='harmlessness',
            example=expert_prompt_examples['harmlessness']
            )

        # Initialise Overseer
        self.overseer = Overseer(self.config['helpfulness_thresh'], self.config['harmlessness_thresh'])

        # # Initialise Teacher
        # teacher = Teacher()
        
        self.helpfulness_ratings = []
        self.harmlessness_ratings = []
        self.has_run = False
    
    def run(self):
        """Iterate through all system steps."""

        for step in range(self.config['num_steps']):

            print(f"Step {step+1}/{self.config['num_steps']}:")

            # Get user query
            query = self.user.get_query()

            # Get worker response
            response = self.worker.generate_text([query])[0]
            print(response)

            # Get helpfulness rating
            helpfulness_rating = self.helpfulness_expert.rate_responses([query], [response])[0]
            self.helpfulness_ratings.append(helpfulness_rating)

            # Get harmlessness rating
            harmlessness_rating = self.harmlessness_expert.rate_responses([query], [response])[0]
            self.harmlessness_ratings.append(harmlessness_rating)

            if (step + 1) % self.config['overseer_steps'] == 0:
                print('\nOverseer evaluating model...')
                helpful, harmless = self.overseer.evaluate(helpfulness_rating, harmlessness_rating)
                if helpful == False:
                    print('Helpfulness threshold not met, sending worker for retraining.')
                #     worker = teacher.teach(worker, 'helpfulness')
                else:
                    print('Helpfulness evaluation passed.')
                if harmless == False:
                    print('Harmlessness threshold not met, sending worker for retraining.')
                #     worker = teacher.teach(worker, 'harmlessness')
                else:
                    print('Harmlessness evaluation passed.')

            print(f"\nHelpfulness: {helpfulness_rating}/10")
            print(f"Harmlessness: {harmlessness_rating}/10.\n")
            
            self.has_run = True

    def evaluate(self):
        """Print key statistics from system run."""

        if self.has_run == False:
            raise Exception('System has not been run yet.')

        print(f"Mean helpfulness = {np.mean(self.helpfulness_ratings)}/10")
        print(f"Mean harmlessness = {np.mean(self.harmlessness_ratings)}/10")
        print(f"Helpfulness failures = {self.overseer.helpfulness_failure_count}/{self.config['num_steps']} steps")
        print(f"Harmlessness failures = {self.overseer.harmlessness_failure_count}/{self.config['num_steps']} steps")
        # print(f"Total training time = {teacher.training_time:.2f}s/{worker.up_time:.2f}s total worker up time.")
