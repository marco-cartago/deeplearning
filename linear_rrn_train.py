import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src.models import TextRNN


class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y



def generate(
    model: nn.Module, 
    int_to_char: dict[int, str], 
    char_to_int: dict[str, int], 
    start_str:str = " ", 
    length: int = 64, 
    temperature:float =1.0
) -> str:
    """
    Given a sequence model iteratively generates a sequence from the model.
    Assumes that the model outputs (y, m) = model(x) where y are the outputs
    for a particular item in the sequence. 
    """
    model.eval()
    chars_out = list(start_str)
    input_seq = torch.tensor([char_to_int[c] for c in start_str]).unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            logits, _ = model(input_seq)
            last_logit = logits[0, -1, :] / temperature
            
            # Sample from the distribution
            probs = torch.softmax(last_logit, dim=-1)
            next_char_id = torch.multinomial(probs, 1).item()
            
            chars_out.append(int_to_char[next_char_id])
            input_seq = torch.cat(
                [input_seq[:, 1:], torch.tensor([[next_char_id]])], 
                dim=1
            )
            
    return "".join(chars_out)



def main():
    with open("dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()

    seq_len     = 64
    batch_size  = 32 
    embed_size  = 30 
    hidden_size = 64

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long)

    dataset = TextDataset(data, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TextRNN(vocab_size, embed_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(25):
        model.train()
        total_loss = 0
        for (x, y) in tqdm(loader):
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\"{generate(model, int_to_char, char_to_int, start_str=".")}\"")
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
   
if __name__ == "__main__":
    main()