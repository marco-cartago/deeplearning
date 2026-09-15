from collections import Counter
import pickle
import re

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from mamba import MambaForCausalLM
from lmtools import preprocess_data, get_charset
from testing import ultrasmart_generate_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_mamba_model(pkl_path, pth_path):
    with open(pkl_path, "rb") as file:
        mlog = pickle.load(file)
        
    config = mlog.model_config
    model = MambaForCausalLM(config)
    state_dict = torch.load(pth_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, mlog


def clean_and_tokenize(text: str, use_words: bool = True):
    if use_words:
        text = text.lower()
        words = re.findall(r'\b[a-zàèéìòù]+\b', text)
        return words
    else:
        text = text.lower()
        transl_table = text.maketrans("«»“”‘’", "\"\"\"\"''")
        text = text.translate(transl_table)
        return list(text)


def get_ngrams(words: list[str], n: int, use_words: bool = True):
    if use_words:
        return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    else:
        return [''.join(words[i:i+n]) for i in range(len(words)-n+1)]


def experiment_ngrams(original_text_path, gen_text_path, n=3, 
                      use_words: bool = True, save_path="./figures/ngrams.png"):
    print(f"\nCheck {n}-gram overlap between original and generated text")

    with open(original_text_path, 'r', encoding='utf-8') as f:
        original_words = clean_and_tokenize(f.read(), use_words=use_words)

    with open(gen_text_path, 'r', encoding='utf-8') as f:
        generated_words = clean_and_tokenize(f.read(), use_words=use_words)

    orig_ngrams = Counter(get_ngrams(original_words, n, use_words=use_words))
    gen_ngrams = Counter(get_ngrams(generated_words, n, use_words=use_words))
    
    print(f"\nTop 5 {n}-grams in Divina Commedia:")
    for gram, count in orig_ngrams.most_common(5):
        print(f"  '{gram}' : {count} times")
        
    print(f"\nTop 5 {n}-grams in Generated Text:")
    for gram, count in gen_ngrams.most_common(5):
        print(f"  '{gram}' : {count} times")

    overlap = set([g for g, c in gen_ngrams.most_common(50)]) & set([g for g, c in orig_ngrams.most_common(50)])
    print(f"\nOverlap in Top 50 {n}-grams: {len(overlap)}/50")

    if not use_words:
        top_12_orig = orig_ngrams.most_common(12)
        keys = [k.replace(' ', '_').replace('\n', '\\n') for k, _ in top_12_orig]
        
        total_orig = orig_ngrams.total()
        total_gen = gen_ngrams.total()
        
        # Relative frequency per 1,000 n-grams for the top 12 original keys
        orig_values_pct = [1000 * count / total_orig for _, count in top_12_orig]
        gen_values_pct = [1000 * gen_ngrams[gram] / total_gen for gram, _ in top_12_orig]

        # Plotting paired horizontal bars
        y = np.arange(len(keys))
        height = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(y - height/2 - 0.01, orig_values_pct, height, label='Original Text', color='tab:blue')
        ax.bar(y + height/2 + 0.01, gen_values_pct, height, label='Generated Text', color='tab:orange')

        ax.set_xticks(y)
        ax.set_xticklabels(keys)
        # ax.invert_yaxis()  # Highest frequency trigram at the top
        ax.set_xlabel(f'Frequency per 1,000 {n}-grams')
        ax.set_title(f'Top 12 {n}-Grams Comparison (Original vs. Generated)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)




def experiment_zipf(original_text_path, gen_text_path, save_path="./figures/zipf.png"):
    """Plots the word frequency distribution on a log-log scale."""
    print("\nCheck word frequency distribution")
    
    with open(original_text_path, 'r', encoding='utf-8') as f:
        original_words = clean_and_tokenize(f.read())

    with open(gen_text_path, 'r', encoding='utf-8') as f:
        generated_words = clean_and_tokenize(f.read())

    orig_counts = sorted(list(Counter(original_words).values()), reverse=True)
    gen_counts = sorted(list(Counter(generated_words).values()), reverse=True)

    tot_orig_counts = sum(orig_counts)
    tot_gen_counts = sum(gen_counts)
    orig_counts = list(map(lambda x: x / tot_orig_counts, orig_counts))
    gen_counts = list(map(lambda x: x / tot_gen_counts, gen_counts))


    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(orig_counts) + 1), orig_counts, label='Divina Commedia', color='blue', linewidth=2)
    plt.loglog(range(1, len(gen_counts) + 1), gen_counts, label='Model Generated Text', color='orange', linewidth=2, linestyle='--')
    
    plt.title("Word Frequency Distribution\n(Log-Log Scale)")
    plt.xlabel("Rank of word")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(save_path)


def calculate_loss_on_text(model, config, text_path, context_len = 512):
    """Calculates the average Cross-Entropy loss of the model on a given text file."""
        
    tokens = get_charset(config.charset_file)
    data = preprocess_data(text_path, tokens).to(DEVICE)
    
    model.eval()
    total_loss = 0.0
    num_chunks = 0
    
    with torch.no_grad():
        for i in range(0, len(data) - context_len, context_len):
            chunk = data[i : i + context_len + 1]
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:]
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y)
            total_loss += loss.item()
            num_chunks += 1
        
    return total_loss / num_chunks


def experiment_domain_specialization(model, config):
    print("\nEvaluate how much the model is surprised by texts from different domains")
    
    test_files = {
        "Petrarca": "./data/petrarca.txt",
        "Wikipedia": "./data/wikipedia.txt",
    #    "Generated": "./data/experiment_large_genv2.txt",
    #    "Divina": "./data/divina_commedia.txt"
    }
    
    for label, path in test_files.items():
        loss = calculate_loss_on_text(model, config, path, context_len=512)
        if loss is not None:
            perplexity = torch.exp(torch.tensor(loss)).item()
            print(f"  > {label:<35} | Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    PKL_FILE = "./logs/mamba-D512-E2.5-N16-d8_cartago.pkl"
    PTH_FILE = "./pretrained/mamba-D512-E2.5-N16-d8_cartago.pth"
    DANTE_TEXT_FILE = "data/divina_commedia.txt"
    LARGE_GENERATED_FILE_1 = "data/experiment_large_genv1.txt"
    LARGE_GENERATED_FILE_2 = "data/experiment_large_genv2.txt"

    model, mlog = load_mamba_model(PKL_FILE, PTH_FILE)
    print("Loaded model")

    # text = ultrasmart_generate_text(model=model, num_tokens=200_000, temperature=0.5, top_k=10) # remove params for more variance
    # print("Generated text")

    # with open(LARGE_GENERATED_FILE, mode='w') as f:
    #     f.write(text["text"])


    experiment_ngrams(DANTE_TEXT_FILE, LARGE_GENERATED_FILE_1, use_words=False)
    print("Runned n-grams experiment")

    experiment_zipf(DANTE_TEXT_FILE, LARGE_GENERATED_FILE_1)
    print("Runned ZIPF experiment")

    # experiment_domain_specialization(model, mlog.model_config)
    print("Runned domain experiment")