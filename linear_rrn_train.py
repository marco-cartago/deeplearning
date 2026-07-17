import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from rnn_block import LinearRNN

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

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn0 = LinearRNN(embed_size, hidden_size, batch_first=True)
        self.fc0 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.rnn1 = LinearRNN(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None):
        x = self.embedding(x)
        
        out, hn = self.rnn0(x, h0)   # LRNN first pass
        out = self.fc0(out)          # FC in the middle
        out, hn = self.rnn1(out, hn) # LRNN second pass
        
        logits = self.fc1(out)
        return logits, hn

def generate(model, int_to_char, char_to_int, start_str=" ", length=16):
    model.eval()
    chars_out = list(start_str)
    input_seq = torch.tensor([char_to_int[c] for c in start_str]).unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            logits, _ = model(input_seq)
            last_logit = logits[0, -1, :]
            next_char_id = torch.argmax(last_logit).item()
            chars_out.append(int_to_char[next_char_id])
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_char_id]])], dim=1)
            
    return "".join(chars_out)


def main():

    with open("dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()


    seq_len = 16 
    batch_size = 32 
    embed_size = 30 
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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(2):
        model.train()
        total_loss = 0
        for x, y in tqdm(loader):

            optimizer.zero_grad()
            logits, _ = model(x)
            
            # Reshape for CrossEntropyLoss: (batch * seq, vocab)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    print(f"\"{generate(model, int_to_char, char_to_int,  start_str="")}\"")


if __name__ == "__main__":
    main()