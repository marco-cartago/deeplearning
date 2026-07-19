"""
Quick manual test script: load the trained Mamba checkpoint and translate
a handful of example sentences between English and Swahili.

Usage:
    python translator.py
    python translator.py --checkpoint pretrained/mamba5m.pth --max_new_tokens 30
"""
import argparse
import re
import torch
from transformers import AutoTokenizer

from mamba import MambaConfig, MambaForCausalLM

DEFAULT_CHECKPOINT_PATH = "pretrained/mamba5m.pth"

# These must exactly match what was used in mamba_training.py, or the
# checkpoint's weights won't line up with the model's shapes.
MODEL_HYPERPARAMS = dict(
    d_model=256,
    d_state=16,
    expand=2,
    d_conv=4,
    n_layers=8,
    mlp_expand=4,
)


def build_tokenizer():
    """Mirrors the tokenizer setup in mamba_training.py exactly. If you
    change the special tokens or base tokenizer there, update this too -
    otherwise token ids won't match what the checkpoint was trained on."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    special_tokens_dict = {
        "additional_special_tokens": ["[2en]", "[2sw]", "[TRASL]"]
        # "eos_token": "[EOS]",
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Same idea as the heuristic in mamba_training.py: only used to *guess* the
# translation direction when the caller doesn't specify one. It's just a
# handful of marker words, so it's easy to fool - pass --direction
# explicitly whenever you actually know the source language.
SWAHILI_MARKERS = {
    "habari", "paka", "asante", "jina", "ni", "hakuna", "matata",
    "jambo", "karibu", "yako", "langu",
}


def guess_is_swahili(text):
    words = re.findall(r"\w+", text.lower())
    return any(w in SWAHILI_MARKERS for w in words)


def load_model(checkpoint_path, tokenizer, device):
    config = MambaConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_HYPERPARAMS,
    )
    model = MambaForCausalLM(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Be flexible about how the checkpoint was saved: a raw state_dict, or
    # a dict wrapping it under a common key (e.g. if it also stored the
    # optimizer state or epoch number alongside the weights).
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Warning: checkpoint did not match the model exactly.")
        if missing:
            print("  Missing keys:", missing)
        if unexpected:
            print("  Unexpected keys:", unexpected)
        print("  (If this list is long, double-check MODEL_HYPERPARAMS above "
              "matches the config used in mamba_training.py.)")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def translate(model, tokenizer, source_text, device, direction=None,
              max_new_tokens=40, temperature=1.0):
    """
    direction: "en2sw", "sw2en", or None to auto-guess from source_text.
    Greedy decoding, one sentence at a time. Note this re-runs the full
    forward pass on the whole sequence-so-far at every new token, since
    ExplicitSelectiveSSM only implements a full-sequence scan (no
    incremental/cached step function) - fine for a quick manual test,
    but not how you'd want to serve this at scale.
    """
    if direction is None:
        target_tag = "[2en]" if guess_is_swahili(source_text) else "[2sw]"
    else:
        assert direction in ("en2sw", "sw2en"), "direction must be 'en2sw' or 'sw2en'"
        target_tag = "[2sw]" if direction == "en2sw" else "[2en]"

    prompt = f"{target_tag} {source_text} [TRASL]"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        logits = model(generated)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
        next_token_id = torch.argmax(next_token_logits).item()

        if next_token_id == eos_id:
            break

        next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        generated = torch.cat([generated, next_token], dim=1)

    new_ids = generated[0, len(prompt_ids):].tolist()
    translation = tokenizer.decode(new_ids, skip_special_tokens=True)
    return target_tag, translation.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = build_tokenizer()
    model = load_model(args.checkpoint, tokenizer, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params:,} parameters) from {args.checkpoint}\n")

    # A few sentences in both directions, direction auto-guessed like in training.
    test_sentences = [
        "Hakuna matata",
        "The cat is under the table",
        "Habari yako?",
        "Where is the nearest market?",
    ]

    print("--- Translations ---")
    for sentence in test_sentences:
        tag, translation = translate(
            model, tokenizer, sentence, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        direction_label = "SW -> EN" if tag == "[2en]" else "EN -> SW"
        print(f"[{direction_label}] {sentence!r} -> {translation!r}")


if __name__ == "__main__":
    main()
