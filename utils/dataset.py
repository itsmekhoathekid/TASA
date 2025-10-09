import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import librosa
from speechbrain.lobes.features import Fbank
import speechbrain as sb


# [{idx : {encoded_text : Tensor, wav_path : text} }]


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r", encoding= 'utf-8') as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def get_blank_token(self):
        return self.stoi["<blank>"]
    def get_space_token(self):
        return self.stoi["<space>"]
    def __len__(self):
        return len(self.vocab)





class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path, apply_spec_augment=True):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()
        self.apply_spec_augment = apply_spec_augment
        self.fbank = Fbank(
            sample_rate=16000,
            n_mels=80,
            n_fft=512,
            win_length=25,
        )


    def __len__(self):
        return len(self.data)

    def get_fbank(self, waveform, sample_rate=16000):

        # mel_extractor = T.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_fft=512,
        #     win_length=int(0.025 * sample_rate),
        #     hop_length=int(0.010 * sample_rate),
        #     n_mels=80,  
        #     power=2.0
        # )

        # log_mel = mel_extractor(waveform.unsqueeze(0))
        # log_mel = torchaudio.functional.amplitude_to_DB(log_mel, multiplier=10.0, amin=1e-10, db_multiplier=0)

        # return log_mel.squeeze(0).transpose(0, 1)  # [T, 80]
        fbank = self.fbank(waveform)
        return fbank.squeeze(0)  # [T, 80]

    def extract_from_path(self, wave_path):
        sig  = sb.dataio.dataio.read_audio(wave_path)
        return self.get_fbank(sig.unsqueeze(0))

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [[self.eos_token, self.eos_token, self.eos_token]], dtype=torch.long)
        decoder_input = torch.tensor([[self.sos_token, self.sos_token, self.sos_token]] + current_item["encoded_text"], dtype=torch.long)
        tokens = torch.tensor(current_item["encoded_text"], dtype=torch.long)
        fbank = self.extract_from_path(wav_path).float()  # [T, 512]

        return {
            "text": encoded_text,
            "fbank": fbank,
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,
            "tokens": tokens,
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def causal_mask(batch_size, size):
    """Tạo mask cho decoder để tránh nhìn thấy tương lai"""
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, T]

# print(causal_mask(1, 3))

def speech_collate_fn(batch):
    decoder_outputs = [torch.tensor(item["decoder_input"]) for item in batch]
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)
    tokens_lens = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)

    padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)      # [B, T_text]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
    # print(calculate_mask(text_lens, padded_texts.size(1)).shape)
    # print(causal_mask(padded_texts.size(0), padded_texts.size(1)).shape)
    
    text_mask= calculate_mask(text_lens, padded_texts.size(1)).unsqueeze(1) & causal_mask(padded_texts.size(0), padded_texts.size(1))  # [B, T_text, T_text]
    text_mask = text_mask.unsqueeze(1)  # [B, 1, T_text, T_text]
    return {
        "decoder_input": padded_decoder_inputs,
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask,
        "tokens" : padded_tokens,
        "tokens_lens": tokens_lens
    }
