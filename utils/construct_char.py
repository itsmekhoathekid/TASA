import json
import re
def normalize_transcript(text):
    text = text.lower()
    text = re.sub(r"[\'\"(),.!?:]", " ", text)
    text = re.sub(r"\s+", " ", text)  # loại bỏ khoảng trắng dư
    return text.strip()

def load_json(json_path):
    """
    Load a json file and return the content as a dictionary.
    """
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_vocab(json_path):
    data = load_json(json_path)

    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<blank>" : 0
    }

    for idx, item in data.items():
        text = normalize_transcript(item['script'])
        for word in text.strip():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    return vocab

def save_data(data, data_path):
    with open(data_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

import os
def process_data(data_path, vocab, default_data_path, save_path):
    data = load_json(data_path)


    res = []
    for idx, item in data.items():
        
        data_res = {}
        text = normalize_transcript(item['script'])
        unk_id = vocab["<unk>"]
        tokens = [vocab.get(word, unk_id) for word in text.strip()]
        data_res['encoded_text'] = tokens
        data_res['text'] = text
        data_res['wav_path'] = os.path.join(default_data_path, item['voice'])
        res.append(data_res)
    
    save_data(res, save_path)
    print(f"Data saved to {save_path}")


vocab = create_vocab("/home/anhkhoa/transformer_transducer_speeQ/data/train.json")
save_data(vocab, "/home/anhkhoa/transformer_transducer_speeQ/data/vocab_char.json")

process_data("/home/anhkhoa/transformer_transducer_speeQ/data/train.json",
             vocab,
             "/mnt/d/voices/voices",
             "/home/anhkhoa/transformer_transducer_speeQ/data/train_char.json")

process_data("/home/anhkhoa/transformer_transducer_speeQ/data/test.json",
             vocab,
             "/mnt/d/voices/voices",
             "/home/anhkhoa/transformer_transducer_speeQ/data/test_char.json")