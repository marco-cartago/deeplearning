import sys, getopt
import time
from typing import Any

import torch

from mamba import MambaConfig, MambaForCausalLM
from lmtools import get_charset, encode, decode, auto_format, preprocess_data

T = TEMPERATURE = 1.0
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
        # Passaggio forward
        logits = model(seq_token)
        
        # Prendiamo i logit dell'ultimo token della sequenza
        last_token_logits = logits[:, -1, :]
        
        # Calcolo delle probabilità con Softmax
        probs = sf(last_token_logits/temperature)
        
        # Sampling multinomiale per il token successivo
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [1, 1]
        
        # Concateniamo il nuovo token alla sequenza corrente
        seq_token = torch.cat((seq_token, next_token), dim=1)

    # 4. Decodifica della sequenza completa (prompt + token generati)
    generated_ids = seq_token[0].cpu().tolist()
    generated_text = decode(generated_ids, id_to_token)

    print("--- Testo Generato ---")
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
    
    # 1. Inizializziamo le cache per lo stato conv e ssm di tutta la rete
    caches = model.allocate_caches(batch_size, DEVICE)
    
    # 2. Phase 1: PREFILL
    # Elaboriamo il prompt iniziale per popolare le cache con il contesto storico
    for t in range(seq_len - 1):
        input_id = prompt_ids[:, t]
        _, caches = model.step(input_id, caches)
        
    # L'ultimo token del prompt è il punto di partenza per la generazione 
    current_token = prompt_ids[:, -1]
    generated_ids = prompt_ids[0].cpu().tolist()
    
    # 3. Phase 2: GENERAZIONE
    # Ora operiamo con complessità O(1) in memoria e O(1) in tempo, passando solo un token alla volta.
    t0 = time.perf_counter()
    for _ in range(num_tokens):
        # Passaggio forward su un SINGOLO token
        logits, caches = model.step(current_token, caches)
        
        # Calcolo probabilità
        probs = sf(logits / temperature)
        
        # Sampling multinomiale per il token successivo
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [1, 1]
        
        # Aggiungiamo alla lista dei generati e aggiorniamo il token corrente
        generated_ids.append(next_token.item())
        current_token = next_token.squeeze(-1) # [1,]
    t1 = time.perf_counter()
    generated_text = decode(generated_ids, id_to_token)

    return {
        "text": generated_text,
        "elapsed_time": t1-t0
    }

    print("\n--- Testo Generato ---")
    print(generated_text)

if __name__ == '__main__':
    args = sys.argv[1:]
    options = "m:T:L:f:c:"
    long_options = ["model=", "temperature=", "file=", "prompt=",
                     "num_tokens=", "length=", "config="]
    arguments, values = getopt.getopt(args, options, long_options)
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
        else:
            raise getopt.GetoptError(
                f"Unknown argument. Valid arguments are\n"
                f"-c, --config\n"
                f"-f, --prompt, --file\n"
                f"-L, --length, --num_tokens\n"
                f"-m, --model\n"
                f"-T, --temperature\n",
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
    model.load_state_dict(torch.load(PRETRAINED_PATH))
    model.to(DEVICE)

    print("Elaborazione del prompt in corso...")

    output = smart_generate_text(model, PROMPT_FILE, NUM_TOKENS, 
                                 temperature=TEMPERATURE)

    print("\n---Generated text---")
    print(output['text'])
    print(f"\nGeneration time: {output['elapsed_time']:.5f}s")
    


    