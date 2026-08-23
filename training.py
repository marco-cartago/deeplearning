import sys
import argparse
import time

import torch
import tqdm

from mamba import MambaConfig, MambaForCausalLM
from lmtools import get_charset, encode, decode, auto_format, preprocess_data
from mlog import ModelLog

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data: torch.Tensor, context_len: int, batch_size: int):
    #select a random starting point
    ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    #select starting point + context_len
    x = torch.stack([data[i:i+context_len] for i in ix])
    #select target
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def train_loop(
    config: MambaConfig, 
    mlog: ModelLog, 
    encoded_data: torch.Tensor, 
    epochs: int,
    batch_size: int,
    context_len: int,
    pretrained_path: str | None
) -> MambaForCausalLM:

    model = MambaForCausalLM(config)

    mlog.n_epochs = n_epochs 
    mlog.batch_size = batch_size

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

    training_start_time = time.time_ns()

    for _ in pbar:

        epoch_start = time.time_ns() # Epoch start -------------------------------

        # training loop
        x, y = get_batch(encoded_data, context_len, batch_size)
        logits = model(x)
        B, T, C = logits.shape

        loss_val = loss(logits.view(B*T, C), y.view(-1))
        # loss_vals[i] = loss_val.item()
        optim.zero_grad()
        loss_val.backward()

        optim.step()
        epoch_end = time.time_ns() # Epoch end -------------------------------

        scheduler.step(metrics=loss_val.item())
        current_lr = scheduler.get_last_lr()[0] # optim.param_groups[0]['lr']
        pbar.set_postfix({"Loss": f"{loss_val.item():.3f}", "lr": f"{current_lr:.2e}"})

        # Logging
        mlog.e_time_elapsed.append(epoch_end - epoch_start)
        mlog.e_train_loss.append(loss_val.detach().item())
        mlog.e_stepsize.append(current_lr)

    training_end_time = time.time_ns()
    mlog.total_time = training_end_time - training_start_time

    return model



if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description="Mamba Model Training and Generation")
        parser.add_argument("-c", "--config", type=str, help="Path to config JSON")
        parser.add_argument("-f", "--file", type=str, help="Path to text file")
        parser.add_argument("-L", "--context_len", type=int, help="Context length", default=512)
        parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=round(1e6/(512*32)))
        parser.add_argument("-B", "--batch", type=int, help="Batch size", default=32)
        parser.add_argument("-p", "--pretrained", type=str, help="Path to pretrained model")
        parser.add_argument("--lr", type=float, help="Learning rate", default=3e-4)
        parser.add_argument("-g", "--generate", action="store_true", help="Enable generation mode", default=False)
        return parser.parse_args()

    args = parse_args()

    # Mapping to constants
    context_len = args.context_len
    config_path = args.config
    n_epochs = args.epochs
    lr = args.lr
    text_file = args.file
    pretrained_path = args.pretrained
    batch_size = args.batch
    generate = args.generate

    if config_path:
        config = MambaConfig.from_config_json(config_path)
    else:
        config = MambaConfig()

    tokens = get_charset(config.charset_file)
    encoded_data = preprocess_data(text_file, tokens)

    tokens = get_charset(config.charset_file)
    encoded_data = preprocess_data(text_file, tokens)

    mlog = ModelLog(config)
    model = train_loop(config, mlog, encoded_data, 
                       n_epochs, batch_size, context_len, 
                       pretrained_path
                    )
    mlog.dump_to_file("./logs")

    model_name = "mamba-D{D}-E{E:.1f}-N{N}-{d}d_{t}"
    model_name = model_name.format(
        D = config.d_model,
        E = config.expand_factor,
        N = config.d_state,
        d = config.n_layers,
        t = mlog.dump_timestamp
    )
    torch.save(model.state_dict(), f"./pretrained/{model_name}.pth")

    # seq_token = preprocess_data('data/prompt.txt', tokens=tokens).unsqueeze(1)
    # sf = torch.nn.Softmax(dim = -1)
    # model.eval()
    # for i in range(len(seq_token) - 1, 200):
    #     logits = model(seq_token)
    #     next_token = torch.multinomial(sf(logits[:,-1,:]), 1).item()
    #     seq_token = torch.cat((seq_token, torch.tensor(next_token).reshape(1,1)), dim=1)
    # id_to_token = {i: el for i, el in enumerate(tokens)}
    # print(decode(seq_token[0].cpu().numpy(), id_to_token))

    if generate:
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
    