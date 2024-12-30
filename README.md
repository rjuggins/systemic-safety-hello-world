# Systemic Safety Hello World

This is really simple first go at testing an AI system for unexpected failure modes. It is not finished as I did not have access to sufficient compute.

## Get started

Seeing as this code is not finished, you can't actually run it all the way through (in particular, the re-training steps will not happen). Roughly speaking, the way it would work is as follows:

- Download [instruction-tuning](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and [helpfulness and harmlessness](https://github.com/anthropics/hh-rlhf) data, and save in the directory `./data/`
- Create a directory called `./keys/` and save a Hugging Face read and write access token in a file called `hf_key.txt`, then an OpenAI API key in a file called `openai_key.txt`
- Make sure `./config/instructor_config.yaml` is configured in a way that matches your model and filenames, modify the training parameters if you think they can be improved, and set the `save_steps` and `checkpoint_name` parameters to ensure the right checkpoint is pushed to the Hugging Face hub
- Run `python instruction_tuning.py` to instruction tune your model and push it to the Hugging Face hub
- Make sure `./config/config.yaml` and `./config/teacher_config.yaml` contain the right filenames, save steps, and checkpoint names, and modify the training parameters if you wish
- Set your system parameters in `./config/config.yaml`: 
    - `num_steps`: Total number of system iterations
    - `overseer_steps`: Number of steps between each *Overseer* intercept
    - `helpfulness_thresh`: Rating threshold below which (inclusive) the *Worker* is sent for helpfulness retraining
    - `harmlessness_thresh`: Rating threshold below which (inclusive) the *Worker* is sent for harmlessness retraining
- Run `python run_system.py`
- View measurements of system performance (not many are implemented)

## Documentation

For a more detailed explanation of what I was trying to achieve, please read my [Substack post]().

### Project goals

The idea was to construct a system of interacting components and then sweep through various parameters to identify phases of behaviour. These phases could then be differentiated by whether the system behaviour is good or bad and whether it is surprising.

![System phases](./images/unknown_unknowns.png)

### Planned system structure

The system is centred around a *Worker* LLM, and also contains a *User*, an *Overseer*, an *Outside Expert*, and a *Teacher*. The *Outside Expert* is also an LLM, whereas the others are generic bits of code (I wanted to make the *User* an LLM too but didn’t get that far). The operation of the system was intended to run as follows:

1. *User* sends a query to the *Worker*
2. *Worker* responds to query
3. *Overseer* periodically intercepts response and sends to two *Outside Experts*
4. *Outside Experts* rate the response for helpfulness and harmlessness and send ratings back to *Overseer*
5. *Overseer* checks if either rating is above a re-training threshold
6. If ratings high enough, return to step 1
7. If either rating below threshold, *Worker* is sent to the *Teacher* for re-training in that domain, and then return to step 1

![System diagram](./images/systemic_safety_hello_world.png)
*Blue boxes are LLMs, purple are other bits of code, green are strings passed between components. The User was intended to be an LLM, but I didn’t that far and instead just implemented a class that samples from a dataset of queries. Solid lines represent actions that happen at every step, dashed happen conditionally (in the case of the Response and the User, this would only happen if it were actually an LLM).*

There are two other components not part of regular system operation: 

- *Instructor*: Class that instruction-tunes the Worker at the start, so that it would be a capable enough chatbot
- *Perturbation*: This was unimplemented, but the idea was to have another component that could disturb the system and see if it would knock it into some new phases of behaviour. The simplest way of doing this may actually have been to modify the User to use jailbreaking techniques

## Implementation differences

The original idea was that querying the Experts would be done only during a periodic Overseer intercept, because in theory for generic AI systems such calls could be expensive. In this particular case however, as they were cheap and it would give me better visibility of the system, I set it up to query the Experts after every response. The Overseer’s actually implemented function is thus just to periodically check the ratings. 

## Support

If you are interested in any of this and have questions, please email [Richard Juggins](mailto:richard.juggins@gmail.com). If you would like to leave anonymous feedback, please use my [feedback form][def].

[def]: https://docs.google.com/forms/d/e/1FAIpQLSdyisSOndK1H1JT0NAbnA35LJgoJrl9f_NiJi1FEljCr7-kJg/viewform