#!/usr/bin/env python3

"""Script to instruction tune a model."""

import yaml
import torch
from components.instructor import Instructor

if __name__ == "__main__":
    # Load config file into dictionary
    with open("./config/instructor_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print(f"Instruction tuning with parameters: {config}")

    torch.manual_seed(config["seed"])

    instructor = Instructor(config)
    instructor.load_model()
    instructor.load_data(test_size=config["test_frac"])
    instructor.train_model()
    instructor.push_model()
