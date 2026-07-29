import torch

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

def train(file: str = "data/the_little_prince.txt"):
    with open(file, "r", encoding="utf-8") as f:
        raw_data = f.read()
        raw_data = raw_data.replace('\n\n', '\n')
    tokens = sorted(set(raw_data))
    token_to_id = {el: i for i, el in enumerate(tokens)}
    id_to_token = {i: el for i, el in enumerate(tokens)}
    encode = lambda x: torch.tensor([token_to_id[s] for s in x])
    decode = lambda x: "".join([id_to_token[s] for s in x])
    encoded_data = encode(raw_data)
    len_data = len(encoded_data)
    train_data = encoded_data[:int(len_data*0.9)]
    test_data = encoded_data[int(len_data*0.9):]

if __name__ == '__main__':
    train()