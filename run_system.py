#!/usr/bin/env python3

import os
import yaml

from models.worker import Worker
from models.outside_expert import OutsideExpert
from models.user import User


def main():
    """Runs AI system end-to-end."""

    # Load config file into dictionary
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialise models
    
    # Initialise Worker
    worker = Worker(config['model_id'])
    worker.check_device_map(no_split_module_classes=config['no_split_module_classes'])
    worker.load_model()
    with open('./config/worker_prompt_examples.yaml', 'r') as file:
        worker_prompt_examples = yaml.safe_load(file)
    worker.set_prompt_examples(worker_prompt_examples)

    # Initialise User
    user = User(data_path=config['user_data_path'])

    # Initialise OutsideExperts
    with open(config['openai_key_path'], 'r') as file:
        openai_key = file.read().strip()
    with open('./config/expert_prompt_examples.yaml', 'r') as file:
        expert_prompt_examples = yaml.safe_load(file)

    helpfulness_expert = OutsideExpert(
        model_id=config['expert_model_id'],
        openai_key=openai_key,
        target='helpfulness',
        example=expert_prompt_examples['helpfulness']
        )

    harmlessness_expert = OutsideExpert(
        model_id=config['expert_model_id'],
        openai_key=openai_key,
        target='harmlessness',
        example=expert_prompt_examples['harmlessness']
        )
    
    # Iterate through system steps
    for step in range(config['num_steps']):

        # Get user query
        query = user.get_query()

        # Get worker response
        response = worker.generate_text([query])[0]
        print(response)

        # Get helpfulness rating
        helpfulness_rating = helpfulness_expert.rate_responses([query], [response])[0]

        # Get harmlessness rating
        harmlessness_rating = harmlessness_expert.rate_responses([query], [response])[0]

        print(f"Helpfulness: {helpfulness_rating}/10; Harmlessness: {harmlessness_rating}/10.\n")

if __name__ == "__main__":
    main()
