# Mamba in the Acheron
#### Deep Learning course at University of Trieste (prof. Alessio Ansuini)
by Marco Cartago, Alessandro Longato, Giovanni Zedda 

A.Y. 2025-2026

This repository contains the code of the final project of the course, which consists of an application of state space models for text generation... trained on *La Divina Commedia*, hence the title (❁´◡`❁).

## Setup

Install the libraries in the `requirements.txt` (`torch` and `tqdm`).

## Usage
### Training a Model
Run the script `training.py` with the following flags:
- `-c | --config`: configuration file, mandatory. You can use one from the folder `configs/` or write your own. We suggest using `configs/mamba-131k-v2-config.json` because it is the fastest to train;
- `-p | --pretrained`: pytorch pretrained model path, optional (default: trained from scratch);
- `-f | --file`: document on which you want to execute the training, mandatory;
- `-L | --context_len`: length of the sequence, optional (default 512). Longer sequences might incur into memory issues and slowdowns, shorter are more prone to be less accurate;
- `-B | --batch`: batch size, optional (default 32);
- `-e | --epochs`: number of iterations on groups of `L*B` tokens, optional (default 61);
- `--lr`: learning rate, optional (default 3e-4). It decreases by 25% on plateaues with patience 10 iterations.

### Testing a Model
Run the script `testing.py` with these flags:
- `-c | --config`: configuration file of an existing model, mandatory;
- `-m | --model`: path of a pretrained model, mandatory;
- `-f | --file | --prompt`: give an input to the model with this flag, default `data/prompt.txt`;
- `-L | --length | --num_tokens`: how many tokens to generate, optional (default 100);
- `-T | --temperature`: the generation temperature, optional positive number (default 1.0). The higher $T$ the more random the output, the lower the more deterministic: $$\text{prob}_{i-\text{th token}}=\frac{\exp(\frac{\text{out}_i}{T})}{\sum_{j=1}^{\text{no. tokens}}\exp(\frac{\text{out}_j}{T})}.$$


## AI Policy and Disclaimers

The blocks of code relative to the Mamba implementation have been initially written with the help of Gemini Flash (of course they were reviewed). Later it was explicitely asked to create the `step` method to make use of the state space for inference (otherwise the utility of the model gets blessed). The rest of the code is purely human-made (apart some small snippets inside `testing.py`), with some inspiration from [this repository](https://github.com/nickplas/Intro_to_ML_25-26/blob/main/notebooks/Lab-18-PicoGPT.ipynb). Project idea and test ideas are all the authors'.


## License and Credits
The code is under [MIT License](LICENSE).

Data hereby provided are under their respective licenses and ownerships.

---
Marco Cartago, Alessandro Longato and Giovanni Zedda

MSc students in Data Science and Artificial Intelligence 

University of Trieste, July 2026

---
Copyright (c) 2026 M. Cartago, A. Longato, G. Zedda