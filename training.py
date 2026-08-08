import sys, getopt
import string

import torch
import tqdm

from mamba import MambaConfig, MambaForCausalLM
from lmtools import get_charset, encode, decode, auto_format, preprocess_data

CONFIG: str | None = None

L = CONTEXT_LEN = 512
BATCH_SIZE = 32
EPOCHS = round(1e6/(512*32)) 
    # process ~1M tokens per book by default
    # for a 100k characters text, it is roughly equivalent
    # to 6 actual epochs.
lr = 3e-4
PRETRAINED_PATH: str | None = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data: torch.Tensor, context_len: int, batch_size: int):
    #select a random starting point
    ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    #select starting point + context_len
    x = torch.stack([data[i:i+L] for i in ix])
    #select target
    y = torch.stack([data[i+1:i+L+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def train_loop(config: MambaConfig, encoded_data: torch.Tensor, 
               epochs=EPOCHS,
               batch_size=BATCH_SIZE,
               pretrained_path: str | None = PRETRAINED_PATH) -> MambaForCausalLM:
    model = MambaForCausalLM(config)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, "min", 
        patience=5,
        factor=0.75,
        min_lr=5e-7, 
        threshold=5e-3, 
        cooldown=2
    )

    model.to(DEVICE)
    model.train()
    # loss_vals = [0.]*epochs
    pbar = tqdm.trange(epochs, leave=True, dynamic_ncols=True)
    for i in pbar:
        # training loop
        x, y = get_batch(encoded_data, CONTEXT_LEN, batch_size)
        logits = model(x)
        B, T, C = logits.shape
        loss_val = loss(logits.view(B*T, C), y.view(-1))
        # loss_vals[i] = loss_val.item()
        optim.zero_grad()
        loss_val.backward()
        optim.step()
        scheduler.step(metrics=loss_val.item())
        current_lr = scheduler.get_last_lr()[0] # optim.param_groups[0]['lr']
        pbar.set_postfix({"Loss": f"{loss_val.item():.3f}", "lr": f"{current_lr:.2e}"})
    return model



if __name__ == '__main__':
    args = sys.argv[1:]
    options = "L:B:c:e:f:p:"
    long_options = ["config=", "epochs=", "file=", "batch=", "context_len=", "lr=", "pretrained="]
    arguments, values = getopt.getopt(args, options, long_options)
    for arg, val in arguments:
        if arg in ('-L', "--context_len"):
            L = CONTEXT_LEN = int(val)
        elif arg in ('-c', "--config"):
            CONFIG = val
        elif arg in ('-e', "--epochs"):
            EPOCHS = int(val)
        elif arg in ("--lr"):
            lr = float(val)
        elif arg in ('-f', "--file"):
            text_file = val
        elif arg in ('-p', "--pretrained"):
            PRETRAINED_PATH = val
        elif arg in ('-B', "--batch"):
            BATCH_SIZE = int(val)
        else:
            raise getopt.GetoptError(
                f"Unknown argument. Valid arguments are\n"
                f"-c, --config\n"
                f"-f, --file\n"
                f"-L, --context_len\n"
                f"-e, --epochs\n"
                f"-B, --batch\n"
                f"--lr",
                opt=str(val)
            )
        
    if CONFIG:
        config = MambaConfig.from_config_json(CONFIG)
    else:
        config = MambaConfig()

    tokens = get_charset(config.charset_file)
    encoded_data = preprocess_data(text_file, tokens)

    model = train_loop(config, encoded_data, EPOCHS, BATCH_SIZE, PRETRAINED_PATH)

    model_name = "mamba-D{D}-E{E:.1f}-N{N}-d{d}"
    model_name = model_name.format(
        D = config.d_model,
        E = config.expand_factor,
        N = config.d_state,
        d = config.n_layers,
    )
    torch.save(model.state_dict(), f"pretrained/{model_name}.pth")

    # seq_token = preprocess_data('data/prompt.txt', tokens=tokens).unsqueeze(1)
    # sf = torch.nn.Softmax(dim = -1)
    # model.eval()
    # for i in range(len(seq_token) - 1, 200):
    #     logits = model(seq_token)
    #     next_token = torch.multinomial(sf(logits[:,-1,:]), 1).item()
    #     seq_token = torch.cat((seq_token, torch.tensor(next_token).reshape(1,1)), dim=1)
    # id_to_token = {i: el for i, el in enumerate(tokens)}
    # print(decode(seq_token[0].cpu().numpy(), id_to_token))

    # 2. Caricamento del prompt e preparazione del dizionario
    # preprocess_data restituisce un Tensor 1D -> aggiungiamo la dimensione del batch [1, seq_len]
    prompt_tensor = preprocess_data('data/prompt.txt', tokens=tokens)
    seq_token = prompt_tensor.unsqueeze(0).to(DEVICE)  # Shape: [1, seq_len]

    id_to_token = {i: el for i, el in enumerate(tokens)}
    sf = torch.nn.Softmax(dim=-1)

    # 3. Quanti nuovi token generare (es. 200 token)
    GENERATE_TOKENS = 128

    model.eval()
    with torch.no_grad():  # Disabilita il tracciamento dei gradienti durante l'inferenza
        for _ in range(GENERATE_TOKENS):
            # Passaggio forward
            logits = model(seq_token)
            
            # Prendiamo i logit dell'ultimo token della sequenza
            last_token_logits = logits[:, -1, :]
            
            # Calcolo delle probabilità con Softmax
            probs = sf(last_token_logits)
            
            # Sampling multinomiale per il token successivo
            next_token = torch.multinomial(probs, num_samples=1)  # Shape: [1, 1]
            
            # Concateniamo il nuovo token alla sequenza corrente
            seq_token = torch.cat((seq_token, next_token), dim=1)

    # 4. Decodifica della sequenza completa (prompt + token generati)
    generated_ids = seq_token[0].cpu().tolist()
    generated_text = decode(generated_ids, id_to_token)

    print("--- Testo Generato ---")
    print(generated_text)
    