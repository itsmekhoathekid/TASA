import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models import Transformer
from tqdm import tqdm
import argparse
import yaml
import os 
from utils import logg, causal_mask
from jiwer import wer, cer


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, vocab_len: int, device: torch.device, epoch : int) -> Transformer:
    checkpoint_path = os.path.join(
        config['training']['save_path'],
        f"{config['model']['model_name']}_epoch_{epoch}"
    )
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = Transformer(
        config = config['model'],
        vocab_size=vocab_len
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

class GreedyPredictor:
    def __init__(self, model, vocab, device, max_len=100):
        self.model = model
        self.sos = vocab.get_sos_token()
        self.eos = vocab.get_eos_token()
        self.blank = vocab.get_blank_token()
        self.pad = vocab.get_pad_token()
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
    @torch.no_grad()
    def greedy_decode(self, src, src_mask):
        B = src.size(0)
        
        # Encode
        enc_out, src_mask = self.model.encode(src, src_mask)

        # Init decoder input: [B, 1]
        decoder_input = torch.full((B, 1), self.sos, dtype=torch.long, device=self.device)

        # Track finished sequences
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(self.max_len):
            tgt_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)
            dec_out = self.model.decode(decoder_input, enc_out, src_mask, tgt_mask)

            # logits for next token: [B, vocab]
            logits = dec_out[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)  # [B]

            # Replace next_tokens with BLANK for finished ones (force pad)
            next_tokens = next_tokens.masked_fill(finished, self.blank)

            # Update finished mask
            finished |= (next_tokens == self.eos)

            # only append tokens that are not {% sos, blank, eos %}
            ignore = torch.tensor([self.sos, self.blank, self.eos], device=self.device)
            append_mask = ~torch.isin(next_tokens, ignore)

            # next tokens to append: [B, 1]
            to_append = next_tokens[append_mask].unsqueeze(1)
            if to_append.size(0) > 0:  # some batch samples still valid
                # create empty placeholder and fill only positions to append
                pad_col = torch.zeros(B, 1, dtype=torch.long, device=self.device)
                pad_col[append_mask] = to_append
                decoder_input = torch.cat([decoder_input, pad_col], dim=1)

            if finished.all(): break
        
        return decoder_input.cpu().numpy()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch to load the model from")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Speech2Text(
        json_path=config['training']['train_path'],
        vocab_path=config['training']['vocab_path']
    )
    test_dataset = Speech2Text(
        json_path=config['training']['test_path'],
        vocab_path=config['training']['vocab_path']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training'].get('batch_size', 16),
        shuffle=False,
        collate_fn=speech_collate_fn
    )
    vocab = test_dataset.vocab.stoi
    vocab_len = len(vocab)

    model = load_model(config, vocab_len, device, epoch = args.epoch)

    predictor = GreedyPredictor(model, train_dataset.vocab, device)
    all_gold_texts = []
    all_predicted_texts = []
    result_path = config['training']['result']
    with open(result_path, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                src = batch['fbank'].to(device)
                src_mask = batch['fbank_mask'].to(device)
                tokens = batch["tokens"].to(device)
                predicted_tokens = predictor.greedy_decode(src, src_mask)

                batch_size = src.size(0)

                for batch_idx in range(batch_size):

                    predicted_tokens_clean = [
                        token for token in predicted_tokens[batch_idx]
                        if token != predictor.sos and token != predictor.eos and token != predictor.blank and token != predictor.pad 
                    ]
                    predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]
        
                    tokens_cpu = tokens.cpu().tolist() 
                    gold_text = [predictor.tokenizer[token] for token in tokens_cpu[batch_idx] if token != predictor.blank and token != predictor.pad]
                    gold_text_str = ' '.join(gold_text)

                    predicted_text_str = ' '.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])

                    if config['training']['type'] == "phoneme":
                        predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                        space_token = vocab.get("<space>")
                        predicted_text_str = predicted_text_str.replace(predictor.tokenizer[space_token], ' ')

                        gold_text = ''.join([predictor.tokenizer[token] for token in tokens_cpu[batch_idx] if token != predictor.blank])
                        gold_text_str = gold_text.replace(predictor.tokenizer[space_token], ' ')
                    elif config['training']['type'] == "char":
                        predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                        gold_text_str = ''.join([predictor.tokenizer[token] for token in tokens_cpu[batch_idx] if token != predictor.blank])
                    
                    
                    
                    all_gold_texts.append(gold_text_str)
                    all_predicted_texts.append(predicted_text_str)
                    print("Predicted text: ", predicted_text_str)
                    print("Gold Text: ", gold_text_str)
                    
                    wer_score = wer(gold_text_str, predicted_text_str)
                    cer_score = cer(gold_text_str, predicted_text_str)
                    print(f"WER: {wer_score:.4f}, CER: {cer_score:.4f}")
                    
                    # Ghi ra file
                    f_out.write(f"Gold: {gold_text_str}\n")
                    f_out.write(f"Pred: {predicted_text_str}\n")
                    f_out.write(f"WER: {wer_score:.4f}, CER: {cer_score:.4f}\n")
                    f_out.write("="*50 + "\n")
            
            total_wer = wer(all_gold_texts, all_predicted_texts)
            total_cer = cer(all_gold_texts, all_predicted_texts)
            
            print(f"Total WER: {total_wer:.4f}")
            print(f"Total CER: {total_cer:.4f}")
            f_out.write(f"\n=== Tổng kết ===\n")
            f_out.write(f"Total WER: {total_wer:.4f}\n")
            f_out.write(f"Total CER: {total_cer:.4f}\n")


if __name__ == "__main__":
    main()
