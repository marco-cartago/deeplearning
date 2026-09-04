import sys, getopt
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from mamba import MambaConfig, MambaForCausalLM
from lmtools import get_charset, encode, decode, auto_format, preprocess_data

T = TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 1.0
REPETITION_PENALTY = 1.0

CONFIG: str | None = None
PRETRAINED_PATH: str | None = None
NUM_TOKENS: int = 100

PROMPT_FILE: str = 'data/prompt.txt'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def generate_text(model: MambaForCausalLM, prompt_file: str = 'data/prompt.txt', 
                  num_tokens: int = 100, temperature: float = 1.0):
    if temperature < 1e-8:
        raise ValueError("Argument `temperature` must be strictly positive.")
    tokens = get_charset(config.charset_file)
    # encoded_data = preprocess_data(text_file, tokens)
    prompt_tensor = preprocess_data('data/prompt.txt', tokens=tokens)
    seq_token = prompt_tensor.unsqueeze(0).to(DEVICE)  # Shape: [1, seq_len]

    id_to_token = {i: el for i, el in enumerate(tokens)}
    sf = torch.nn.Softmax(dim=-1)
    model.eval()
    for _ in range(num_tokens):

        logits = model(seq_token)
        
        # Logits from the last token of the sequence
        last_token_logits = logits[:, -1, :]
        
        probs = sf(last_token_logits/temperature)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [1, 1]
        
        # Concatenate the new token to the sequence
        seq_token = torch.cat((seq_token, next_token), dim=1)

    # Decode the sequence
    generated_ids = seq_token[0].cpu().tolist()
    generated_text = decode(generated_ids, id_to_token)

    print("--- Generated Text ---")
    print(generated_text)


@torch.no_grad()
def smart_generate_text(model: MambaForCausalLM, prompt_file: str = 'data/prompt.txt', 
                  num_tokens: int = 100, temperature: float = 1.0) -> dict[str, Any]:
    if temperature < 1e-8:
        raise ValueError("Argument `temperature` must be strictly positive.")
    
    tokens = get_charset(model.config.charset_file)
    prompt_tensor = preprocess_data(prompt_file, tokens=tokens)
    prompt_ids = prompt_tensor.unsqueeze(0).to(DEVICE)  # Shape: [1, seq_len]
    
    batch_size, seq_len = prompt_ids.shape
    id_to_token = {i: el for i, el in enumerate(tokens)}
    sf = torch.nn.Softmax(dim=-1)
    
    model.eval()
    
    # 1. Initialize cache for conv state and ssm
    caches = model.allocate_caches(batch_size, DEVICE)
    
    # 2. Phase 1: PREFILL
    for t in range(seq_len - 1):
        input_id = prompt_ids[:, t]
        _, caches = model.step(input_id, caches)
        
    # Last prompt token is the starting point for generation 
    current_token = prompt_ids[:, -1]
    generated_ids = prompt_ids[0].cpu().tolist()
    
    # 3. Phase 2: GENERATION
    t0 = time.perf_counter()
    for _ in range(num_tokens):
        logits, caches = model.step(current_token, caches)

        probs = sf(logits / temperature)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [1, 1]
        generated_ids.append(next_token.item())
        current_token = next_token.squeeze(-1) # [1,]

    t1 = time.perf_counter()
    generated_text = decode(generated_ids, id_to_token)

    return {
        "text": generated_text,
        "elapsed_time": t1-t0
    }


def apply_top_k_top_p(logits: torch.Tensor, 
                      top_k: int = 0, 
                      top_p: float = 0.0, 
                      filter_value: float = -float('Inf')) -> torch.Tensor:
    """
    Filter logits using Top-K and/or Top-P (Nucleus) sampling.
    logits shape: (Batch, vocab_size)
    """
    logits = logits.clone()
    
    # 1. Apply Top-K
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Find the value of the K-th biggest logit
        k_th_value = torch.topk(logits, top_k)[0][..., -1, None]
        # Mask all logits below the found value
        indices_to_remove = logits < k_th_value
        logits[indices_to_remove] = filter_value # logits are set to -Inf, so they have prob=0

    # 2. Apply Top-P (Nucleus)
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Keep at least the first token by shifting the mask 1 to the right
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Restore the original ordering of the mask to remove
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def ultrasmart_generate_text(
    model: MambaForCausalLM, 
    prompt_file: str = 'data/prompt.txt', 
    num_tokens: int = 100, 
    temperature: float = 1.0,
    top_k: int = 20,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0
) -> dict[str, Any]:
    
    if temperature < 1e-8:
        raise ValueError("Argument `temperature` must be strictly positive.")
    
    tokens = get_charset(model.config.charset_file)
    prompt_tensor = preprocess_data(prompt_file, tokens=tokens)
    prompt_ids = prompt_tensor.unsqueeze(0).to(DEVICE)  # Shape: [1, seq_len]
    
    batch_size, seq_len = prompt_ids.shape
    id_to_token = {i: el for i, el in enumerate(tokens)}
    sf = torch.nn.Softmax(dim=-1)
    
    model.eval()
    
    # 1. Initialize cache for conv state and ssm
    caches = model.allocate_caches(batch_size, DEVICE)
    
    # 2. Phase 1: PREFILL
    for t in range(seq_len - 1):
        input_id = prompt_ids[:, t]
        _, caches = model.step(input_id, caches)
        
    # Last prompt token is the starting point for generation 
    current_token = prompt_ids[:, -1]
    generated_ids = prompt_ids[0].cpu().tolist()
    
    # 3. Phase 2: GENERATION
    t0 = time.perf_counter()
    for _ in range(num_tokens):

        logits, caches = model.step(current_token, caches) # Shape: (1, vocab_size)
        
        # # A. Apply repetition penalty if activated (> 1.0)
        if repetition_penalty != 1.0:
            recent_tokens = set(generated_ids[-4:]) # Look at last 4 genearated tokens
            for token_id in recent_tokens:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        # B. Apply temperature
        logits = logits / temperature
        
        # C. Apply top-K and top-P filter
        filtered_logits = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
        
        # D. Softmax + Sampling
        probs = sf(filtered_logits)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [1, 1]
        
        # E. Update
        generated_ids.append(next_token.item())
        current_token = next_token.squeeze(-1) # Shape: [1,]
        
    t1 = time.perf_counter()
    generated_text = decode(generated_ids, id_to_token)

    return {
        "text": generated_text,
        "elapsed_time": t1 - t0
    }



def plot_speed(model: MambaForCausalLM,
               lengths: tuple[int,...], 
               rep: int = 5, 
               scale: str = 'log-log',
               path: str = 'figures/generation_time.png'):
    assert scale in ('log-log', 'lin-lin', 'log-lin', 'lin-log'), "Invalid scale."
    avg_times = []
    stddev_times = []
    for l in lengths:
        times = np.zeros(rep)
        for r in range(rep):
            output = smart_generate_text(model, num_tokens=l)
            times[r] = output['elapsed_time']
        avg_times.append(times.mean())
        stddev_times.append(times.std(ddof=1) if rep > 1 else 0.)

    # plt.plot(lengths, avg_times)
    plt.figure(figsize=(8,5))
    plt.errorbar(lengths, avg_times, yerr=stddev_times, fmt='-o', 
                 capsize=8, c='steelblue', ecolor='lightskyblue', ms=8)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Elapsed Time')
    scale_x, scale_y = scale.split('-')
    if scale_x == 'log':
        plt.semilogx()
    if scale_y == 'log':
        plt.semilogy()
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.title(f"Tokens Generation Time", fontsize=11)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    



if __name__ == '__main__':
    args = sys.argv[1:]
    options = "m:T:L:f:c:s"
    long_options = ["model=", "temperature=", "file=", "prompt=", "top_k=", 
                    "repetition_penalty=", "top_p=", "num_tokens=", "length=", 
                    "config=", "saveplot"]
    arguments, values = getopt.getopt(args, options, long_options)
    save_plot = False
    for arg, val in arguments:
        if arg in ('-T', "--temperature"):
            T = TEMPERATURE = float(val)
        elif arg in ('-c', "--config"):
            CONFIG = val
        elif arg in ('-f', "--file", "--prompt"):
            PROMPT_FILE = val
        elif arg in ('-m', "--model"):
            PRETRAINED_PATH = val
        elif arg in ('-L', "--num_tokens", "--length"):
            NUM_TOKENS = int(val)
        elif arg in ("--top_k",):
            TOP_K = int(val)
        elif arg in ("--top_p",):
            TOP_P = float(val)
        elif arg in ("--repetition_penalty",):
            REPETITION_PENALTY = float(val)
        elif arg in ('-s', '--saveplot'):
            save_plot = True
        else:
            raise getopt.GetoptError(
                f"Unknown argument. Valid arguments are\n"
                f"-c, --config\n"
                f"-f, --prompt, --file\n"
                f"-L, --length, --num_tokens\n"
                f"-m, --model\n"
                f"-T, --temperature\n"
                f"--top_k\n"
                f"--top_p\n"
                f"--repetition_penalty\n"
                f"-s, --saveplot\n", 
                opt=str(val)
            )
        
    if CONFIG:
        config = MambaConfig.from_config_json(CONFIG)
    else:
        raise getopt.GetoptError("Argument -c | --config is mandatory.")

    if not PRETRAINED_PATH:
        raise getopt.GetoptError("Argument -f | --file relative to the"
                                 " position of the pretrained model is mandatory.")

    model = MambaForCausalLM(config)
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)

    print("Prompt processing ongoing...")

    # output = smart_generate_text(model, PROMPT_FILE, NUM_TOKENS, 
    #                              temperature=TEMPERATURE)

    output = ultrasmart_generate_text(
        model, PROMPT_FILE, NUM_TOKENS,
        temperature = TEMPERATURE,
        top_k = TOP_K,
        top_p = TOP_P,
        repetition_penalty = REPETITION_PENALTY
    )

    print("\n---Generated text---")
    print(output['text'])
    print(f"\nGeneration time: {output['elapsed_time']:.5f}s")

    if save_plot:
        plot_speed(model, 
                   (64, 128, 192, 256, 64*5, 64*6, 64*7, 64*8), 
                   rep=10, 
                   scale='lin-lin', 
                   path='figures/generation_time.png')
    


    