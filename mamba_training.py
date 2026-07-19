import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from mamba import Mamba, MambaConfig, MambaForCausalLM
from parser import TSVParser

# 1. Simulated raw data (including swapped directions)
raw_tuples = [
    ("How are you?", "Habari yako?"),           # EN -> SW
    ("The cat is sleeping", "Paka amelala"),    # EN -> SW
    ("Asante sana", "Thank you very much"),     # SW -> EN (Swapped)
    ("Jina langu ni John", "My name is John")   # SW -> EN (Swapped)
]

# 2. Add Special Tokens to your Tokenizer
# (Assuming you trained a BPE tokenizer as discussed, or using an existing one)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased") 

special_tokens = ["[2en]", "[2sw]", "[TRASL]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# Ensure pad token exists (Crucial for variable lengths)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# IMPORTANT: If you pass this to a pre-trained model or your Mamba block, 
# you MUST resize the embedding layer to accommodate the new special tokens.
# model.embedding.weight = nn.Parameter(torch.empty(len(tokenizer), config.d_model))

class TranslationDataset(Dataset):
    def __init__(self, data_tuples):
        self.data = []
        # Basic heuristic: detect direction to apply the correct tag.
        # In a real pipeline, explicitly define the source/target language.
        for text1, text2 in data_tuples:
            self.data.append((text1, text2, '[2sw]'))
            self.data.append((text2, text1, '[2en]'))
    #     for text1, text2 in data_tuples:
    #         if self.is_swahili(text1):
    #             self.data.append((text1, text2, "[2en]"))
    #         else:
    #             self.data.append((text1, text2, "[2sw]"))

    # def is_swahili(self, text):
    #     # Placeholder logic: identify language based on specific words or a classifier
    #     swahili_markers = ["habari", "paka", "asante", "jina", "ni"]
    #     return any(word in text.lower() for word in swahili_markers)
    @classmethod
    def from_tsv(cls, 
                 path: str, 
                 columns: tuple[int, ...] = (1,3),
                 header: bool = False,
                 encoding: None | str = 'UTF-8'):
        tsvparser = TSVParser()
        data_tuples = tsvparser.readfile(path, columns, header, encoding)
        return cls(data_tuples)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def right_pad_and_mask_collate(batch):
    batch_input_ids = []
    batch_labels = []
    max_len = 0
 
    for source_text, target_text, target_tag in batch:
        # 1. Format the string
        prompt = f"{target_tag} {source_text} [TRASL]"
        target = f" {target_text}{tokenizer.eos_token}"
 
        # 2. Tokenize separately to isolate the prompt for masking
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
 
        input_ids = prompt_ids + target_ids
 
        # 3. Mask the prompt with -100 so the model only learns the translation
        labels = [-100] * len(prompt_ids) + target_ids
 
        batch_input_ids.append(torch.tensor(input_ids))
        batch_labels.append(torch.tensor(labels))
 
        # Track max length in this specific batch
        max_len = max(max_len, len(input_ids))
 
    # 4. Right-pad the tensors.
    #
    # This was previously left-padding. That's the right call for batched
    # generation with attention-based models (so every sequence's "last
    # real token" lines up at the same index), but it's the wrong choice
    # for training a state-space model like Mamba. Mamba has no attention
    # mask - every timestep unconditionally updates the recurrent hidden
    # state - so padding placed *before* the real tokens leaks into every
    # real token's state. Padding placed *after* the real tokens only
    # affects positions that come after the real content (which the
    # causal recurrence can't propagate backwards) and whose loss is
    # already masked out with -100, so it's harmless.
    padded_input_ids = []
    padded_labels = []
    pad_id = tokenizer.pad_token_id
 
    for inp, lbl in zip(batch_input_ids, batch_labels):
        pad_len = max_len - len(inp)
 
        padded_inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
        padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
 
        padded_input_ids.append(padded_inp)
        padded_labels.append(padded_lbl)
 
    return torch.stack(padded_input_ids), torch.stack(padded_labels)
 
 
# Initialize DataLoader
dataset = TranslationDataset.from_tsv('data/english-swahili.tsv')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=right_pad_and_mask_collate)
 
 
# Setup device and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = MambaConfig(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
model = MambaForCausalLM(config)
model = model.to(device)
 
# Standard learning rate for custom causal language models
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
 
epochs = 5
 
model.train()
for epoch in range(epochs):
    # Initialize tqdm progress bar for the epoch
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    epoch_loss = 0.0
 
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
 
        optimizer.zero_grad()
 
        # Forward pass (Using the custom Mamba model built previously)
        # Logits shape: (Batch, Seq_Len, Vocab_Size)
        logits = model(input_ids) 
        
        # Shift logits and labels for next-token prediction
        # The model predicts token t+1 using the state at token t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
 
        # Flatten tensors to calculate Cross Entropy
        # (Batch * Seq_Len_Minus_1, Vocab_Size)
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
 
        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping is highly recommended for State Space Models 
        # to prevent explosion in the recurrent state matrix
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
 
        # Update progress bar metrics
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
 
    print(f"End of Epoch {epoch+1} | Average Loss: {epoch_loss / len(dataloader):.4f}")
torch.save(model.state_dict(), 'pretrained/mamba5m.pth')