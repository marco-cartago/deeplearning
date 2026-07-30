import sys, getopt
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from mamba import MambaConfig, MambaForCausalLM
from training import CONTEXT_LEN

L = CONTEXT_LEN = 512
BATCH_SIZE = 32
EPOCHS = round(1e6/(512*32)) 
    # process ~1M tokens per book by default
    # for a 100k characters text, it is roughly equivalent
    # to 6 actual epochs.
lr = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(data: List[int], context_len: int, batch_size: int):
    #select a starting point
    ix = torch.randint(len(data) - context_len - 1, (batch_size,))

    #select starting point + context_len
    x = torch.stack([data[i:i+L] for i in ix])

    #select target
    y = torch.stack([data[i+1:i+L+1] for i in ix])

    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

def train_loop(model, optimizer, epochs, **kwargs):
    pass


if __name__ == '__main__':
    args = sys.argv[1:]
    options = "L:B:c:e:f:"
    long_options = ["config=", "epochs=", "file=", "batch=", "context_len=", "lr="]
    arguments, values = getopt.getopt(args, options, long_options)
    for arg, val in arguments:
        if arg in ('-L', "--context_len"):
            L = int(val)
        elif arg in ('-c', "--config"):
            CONFIG: str = val
        elif arg in ("-e", "--epochs"):
            EPOCHS = val
        elif arg in ("--lr"):
            lr = float(val)

        else:
            raise getopt.GetoptError(
                f"Unknown argument. Valid arguments are\n"
                f"-c, --config\n"
                f"-f, --file\n"
                f"-L, --context_len\n"
                f"-e, --epochs\n"
                f"-B, --batch.",
                opt=str(val)
            )