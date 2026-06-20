import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A custom recurrent neural network
class GRLU(nn.Module):
    
    def __init__(
            self, 
            vocab_size: int,
            num_layers: int, 
            hidden_size: int,
        ):

        super(GRLU, self).__init__()
        
        # Internal dimensions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Hidden unit parameters
        # self.L_att = nn.Linear(hidden_size, hidden_size)
        self.Wz = nn.Linear(hidden_size, hidden_size) 
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Wr = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)
        self.Y = nn.Linear(hidden_size, vocab_size)

    def _cell_step(self, x, h):
        xt = self.embedding(x)
        zt = F.sigmoid(self.Wz(xt) + self.Uz(h))
        rt = F.sigmoid(self.Wr(xt) + self.Ur(h))
        h_hatt = F.tanh(self.Wh(xt) + self.Uh(xt + torch.mul(rt, h)))
        ht = torch.mul((1 - zt), h) + torch.mul(zt, h_hatt)
        y = self.Y(zt)

        return y, ht

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(DEVICE)

    def forward(self, x, hidden):
        # Assumes the batch size is the first one.
        seq_len = x.size(1)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t]                    
            output, hidden = self._cell_step(x_t, hidden)
            outputs.append(output)
        
        out = torch.cat(outputs, dim=0)
        new_hidden = (hidden)
        
        return out, new_hidden


# LSTM model
class CharLSTM(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embed(x)  # Convert to embeddings
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        )

