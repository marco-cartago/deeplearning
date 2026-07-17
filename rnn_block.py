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


class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, batch_first=False):
        super(LinearRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Weights for input-to-hidden and hidden-to-hidden transitions
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self._initialize_w_mat()

        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias', None)

    def _initialize_w_mat(self):
        """Initializes the embedding matrix to be orthogonal, so that the eigenvalues of the transformation are initially one."""
        nn.init.orthogonal_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """Expects x as shape (seq_len, batch, input_size) or (batch, seq_len, input_size)"""
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()
        
        if h0 is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        else:
            h = h0[0] if isinstance(h0, tuple) else h0

        # Linear recurrence
        outputs = []
        for t in range(seq_len):
            h = torch.matmul(x[t], self.weight_ih.t()) + \
                torch.matmul(h[-1], self.weight_hh.t())
            
            if self.bias is not None:
                h = h + self.bias

            outputs.append(h.unsqueeze(0))

        output = torch.cat(outputs, dim=0)
        hn = h.unsqueeze(0)

        if self.batch_first:
            output = output.transpose(0, 1)

        # Return (output, hn)
        return output, hn
