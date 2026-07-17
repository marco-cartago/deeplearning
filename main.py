import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.rnn_block import *

def main():

    # Data I/O
    data = open('dataset.txt', 'r', encoding='utf8').read()
    chars = list(set(data))
    print(f"Found {len(chars)} different characters.")
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    vocab_size = len(chars)
    hidden_size = 1024      # Hidden state size
    num_layers = 1         # Two-layer LSTM
    seq_length = 25        # Sequence length
    learning_rate = 2e-3   # Learning rate
    batch_size = 64        # Batch size

    # Initialize model, loss, and optimizer
    # model = CharRNN(vocab_size, hidden_size, num_layers).to(DEVICE)
    model = GRLU(vocab_size, num_layers, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 5000  # Number of iterations
    hprev = model.init_hidden(batch_size)

    for epoch in range(num_epochs):

        if (epoch * seq_length + seq_length >= len(data)):
            hprev = model.init_hidden(batch_size)  # Reset hidden state
            start = 0
        else:
            start = epoch * seq_length
        
        inputs = []
        targets = []

        for i in range(batch_size):
            start_idx = (epoch * seq_length + i * seq_length) % (len(data) - seq_length)
            input_seq = [char_to_ix[ch] for ch in data[start_idx:start_idx+seq_length]]
            target_seq = [char_to_ix[ch] for ch in data[start_idx+1:start_idx+seq_length+1]]
            
            inputs.append(input_seq)
            targets.append(target_seq)

        inputs = torch.tensor(inputs, dtype=torch.long).to(DEVICE)  # Shape: (batch_size, seq_length)
        targets = torch.tensor(targets, dtype=torch.long).to(DEVICE)  # Shape: (batch_size, seq_length)

        
        model.zero_grad()
        outputs, hprev = model(inputs, hprev)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()

        hprev = model.init_hidden(batch_size)
        

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}',end='\r')
        
        if epoch % 50 == 0:    
            # Sample text
            sample_input = torch.tensor([char_to_ix[data[start]]], dtype=torch.long).unsqueeze(0).to(DEVICE)
            h_sample = model.init_hidden(1)
            sampled_chars = []
            
            for _ in range(300):  # Generate 300 characters
                output, h_sample = model(sample_input, h_sample)
                prob = torch.nn.functional.softmax(output[-1], dim=0).detach().cpu().numpy()
                char_index = np.random.choice(vocab_size, p=prob)

                sampled_chars.append(ix_to_char[char_index])
                sample_input = torch.tensor([[char_index]], dtype=torch.long).to(DEVICE)
            
            print("----\n" + ''.join(sampled_chars) + "\n----")


if __name__ == "__main__":
    main()