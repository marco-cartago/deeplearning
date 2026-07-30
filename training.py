import torch
from tqdm import tqdm, trange

from mamba import MambaConfig, MambaForCausalLM

BATCH_SIZE = 64
CONTEXT_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(data):
    #select a starting point
    ix = torch.randint(len(data) - CONTEXT_LEN, (BATCH_SIZE,))

    #select starting point + context_len
    x = torch.stack([data[i:i+CONTEXT_LEN] for i in ix])

    #select target
    y = torch.stack([data[i+1:i+CONTEXT_LEN+1] for i in ix])

    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def train(doc: str = "data/the_little_prince.txt"):
    with open(doc, "r", encoding="utf-8") as f:
        raw_data = f.read()
        raw_data = raw_data.replace('\n\n', '\n')
    tokens = sorted(set(raw_data))
    globals()['tokens'] = tokens
    token_to_id = {el: i for i, el in enumerate(tokens)}
    id_to_token = {i: el for i, el in enumerate(tokens)}
    encode = lambda x: torch.tensor([token_to_id[s] for s in x])
    decode = lambda x: "".join([id_to_token[s] for s in x])
    encoded_data = encode(raw_data)
    len_data = len(encoded_data)
    train_data = encoded_data[:int(len_data*0.9)]
    test_data = encoded_data[int(len_data*0.9):]
    config = MambaConfig(
        vocab_size=len(tokens),
        d_model=128,
        d_state=8,
        expand_factor=2,
        d_conv=4,
        dt_rank='auto',
        n_layers=8
    )
    model = MambaForCausalLM(config)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr = 3e-4)

    model.to(DEVICE)
    model.train()
    num_iters = 200
    loss_vals = [0. for _ in range(num_iters)]
    pbar = trange(num_iters, leave=True, dynamic_ncols=True)
    for i in pbar:
        # training loop
        x, y = get_batch(train_data)
        logits = model(x)
        B, T, C = logits.shape
        loss_val = loss(logits.view(B*T, C), y.view(-1))
        loss_vals[i] = loss_val.item()
        optim.zero_grad()
        loss_val.backward()
        optim.step()
        pbar.set_postfix({"Loss": f"{loss_val.item():.3f}"})
    return model


if __name__ == '__main__':
    train()