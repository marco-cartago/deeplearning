import string

import torch

def get_charset(charset_file: str | None = None) -> list[str]:
    if not charset_file:
        return sorted(set(string.printable[:-3]))
    
    with open(charset_file, "r", encoding="utf-8") as f:
        raw_data = f.read()
    tokens = sorted(set(raw_data))
    return tokens

def encode(string, token_to_id: dict[str, int]) -> torch.Tensor:
    return torch.tensor([token_to_id[s] for s in string])

def decode(ids, id_to_token: dict[int, str]) -> str:
    return "".join([id_to_token[s] for s in ids])

def auto_format(text: str) -> str:
    """Preprocess text in order to avoid things 
    like double spaces or double newlines"""
    new_text = text
    # new_text = text.replace('  ', ' ')
    # new_text = new_text.replace('\n\n', '\n')
    transl_table = new_text.maketrans("«»“”‘’", "\"\"\"\"''")
    new_text = new_text.translate(transl_table)
    return new_text

def preprocess_data(doc_path: str, 
                tokens: list[str] | None = None) -> torch.Tensor:
    if not tokens:
        tokens = sorted(set(string.printable[:-3]))
    token_to_id = {el: i for i, el in enumerate(tokens)}

    with open(doc_path, "r", encoding="utf-8") as f:
        raw_data = f.read()
    data = auto_format(raw_data)

    return encode(data, token_to_id)

if __name__ == '__main__':
    # import pprint
    tokens = get_charset("dictionaries/latin1-charset.txt")
    token_to_id = {el: i for i, el in enumerate(tokens)}
    id_to_token = {i: el for i, el in enumerate(tokens)}

    print(tokens)
    print("\nToken2ID:")
    # pprint.pp(token_to_id)
    print("Example of encoding: 'Hello, world!' ->", encode("Hello, world!", token_to_id))
    print("\nID2Token:")
    # pprint.pp(id_to_token)
    print("Example of decoding: [33,34,35] ->", decode([33,34,35], id_to_token))



    