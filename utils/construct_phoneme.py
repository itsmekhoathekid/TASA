import json
import re
from word_decomposation import analyse_Vietnamese

def normalize_transcript(text):
    text = text.lower()
    text = re.sub(r"[\'\"(),.!?]", " ", text)
    text = re.sub(r"\s+", " ", text)  # loại bỏ khoảng trắng dư
    return text.strip()

def load_json(json_path):
    """
    Load a json file and return the content as a dictionary.
    """
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_vocab(json_path, wrong2correct):
    unprocsssed = []
    data = load_json(json_path)

    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<space>": 4,
        "<blank>" : 5
    }

    for idx, item in data.items():
        text = normalize_transcript(item['script'])
        for word in text.split():
            try:
                initial, rhyme, tone = analyse_Vietnamese(word)
                if initial not in vocab:
                    vocab[initial] = len(vocab)
                if rhyme not in vocab:
                    vocab[rhyme] = len(vocab)
                if tone not in vocab:
                    vocab[tone] = len(vocab)
            except:
                if word in wrong2correct.keys():
                    correct_word = wrong2correct[word]
                    try:
                        initial, rhyme, tone = analyse_Vietnamese(correct_word)
                        if initial not in vocab:
                            vocab[initial] = len(vocab)
                        if rhyme not in vocab:
                            vocab[rhyme] = len(vocab)
                        if tone not in vocab:
                            vocab[tone] = len(vocab)
                    except:
                        unprocsssed.append(word)
                
    
    return vocab, list(set(unprocsssed))

def save_data(data, data_path):
    with open(data_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

import os
def process_data(data_path, vocab, default_data_path, save_path, type = "stack"):
    data = load_json(data_path)


    res = []
    for idx, item in data.items():
        
        data_res = {}
        text = normalize_transcript(item['script'])
        unk_id = vocab["<unk>"]
        # tokens = [vocab.get(word, unk_id) for word in text.split()]

        tokens = []
        for word in text.split():
            try:
                initial, rhyme, tone = analyse_Vietnamese(word)
                word_list = [vocab.get(initial, unk_id), vocab.get(rhyme, unk_id), vocab.get(tone, unk_id)]
                if type == "stack":
                    tokens.append(word_list)
                else:
                    tokens += word_list
                    tokens += [vocab["<space>"]]
            except:
                continue


        data_res['encoded_text'] = tokens[:-1] if type != "stack" else tokens
        data_res['text'] = text
        data_res['wav_path'] = os.path.join(default_data_path, item['voice'])
        res.append(data_res)
    
    save_data(res, save_path)
    print(f"Data saved to {save_path}")

wrong2correct = {
    "piêu": "phiêu",
    "quỉ": "quỷ",
    "téc": "tét",
    "quoạng": "quạng",
    "đéc": "đét",
    "quĩ": "quỹ",
    "ka": "ca",
    "gen": "ghen",
    "qui": "quy",
    "ngía": "nghía",
    "quít": "quýt",
    "yêng": "yên",
    "séc": "sét",
    "quí": "quý",
    "quị": "quỵ",
    "pa": "ba",
    "ko": "không",
    "léc": "lét",
    "pí": "bí",
    "quì": "quỳ",
    "pin": "bin"
}


# vocab, unprocossed = create_vocab("workspace/dataset/train.json", wrong2correct)
# save_data(vocab, "workspace/dataset/vocab_phoneme.json")

# process_data("workspace/dataset/train.json",
#              vocab,
#              "workspace/dataset/voices",
#              "workspace/dataset/train_phoneme.json")

# process_data("workspace/dataset/test.json",
#              vocab,
#              "workspace/dataset/voices",
#              "workspace/dataset/test_phoneme.json")

# print("Unprocessed words:", unprocossed)


vocab, unprocossed = create_vocab("/home/anhkhoa/transformer_transducer_speeQ/data/train.json", wrong2correct)
save_data(vocab, "/home/anhkhoa/transformer_transducer_speeQ/data/vocab_phoneme_flat.json")

process_data("/home/anhkhoa/transformer_transducer_speeQ/data/train.json",
             vocab,
             "/mnt/d/voices/voices",
             "/home/anhkhoa/transformer_transducer_speeQ/data/train_phoneme_flat.json", "falat")

process_data("/home/anhkhoa/transformer_transducer_speeQ/data/test.json",
             vocab,
             "/mnt/d/voices/voices",
             "/home/anhkhoa/transformer_transducer_speeQ/data/test_phoneme_flat.json", "flat")

print("Unprocessed words:", unprocossed)