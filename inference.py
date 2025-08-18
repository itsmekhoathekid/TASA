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
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
    def greedy_decode(self, src, src_mask):
        enc_out, src_mask = self.model.encode(src, src_mask)
        decoder_input = torch.tensor([[self.sos]], dtype=torch.long).to(self.device)

        for _ in range(self.max_len):
            decoder_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)
            # print("decoder mask : ", decoder_mask.shape)
            # print("enc out shape : ", enc_out.shape)
            dec_out = self.model.decode(decoder_input, enc_out, src_mask, decoder_mask)
            prob = dec_out[:, -1, :]  # [B, vocab_size]

            _, next_token = torch.max(prob, dim=1)  # [B]

            if next_token not in [self.sos, self.eos, self.blank]:
                next_token_tensor = torch.tensor([[next_token.item()]], dtype=torch.long).to(self.device)
                decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

            if next_token == self.eos:
                break
        
        return decoder_input.squeeze(0).cpu().numpy()


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
        batch_size=1,
        shuffle=False,
        collate_fn=speech_collate_fn
    )
    vocab = test_dataset.vocab.stoi
    vocab_len = len(vocab)

    model = load_model(config, vocab_len, device, epoch = args.epoch)

    predictor = GreedyPredictor(model, train_dataset.vocab, device)
    all_gold_texts = []
    all_predicted_texts = []
    result_path = 'workspace/TASA/result.txt'
    with open(result_path, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                src = batch['fbank'].to(device)
                src_mask = batch['fbank_mask'].to(device)
                tokens = batch["tokens"].to(device)
                # print("src shape : ", src.shape)
                # print("src mask : ", src_mask.shape)
    
                predicted_tokens = predictor.greedy_decode(src, src_mask)
                # predicted_text = [predictor.tokenizer[token] for token in predicted_tokens]
                # print("Predicted text: ", ' '.join(predicted_text))
                # tokens_cpu = tokens.cpu().tolist() 
                # gold_text = [predictor.tokenizer[token] for token in tokens_cpu[0]]
                # print("Gold Text: ", ' '.join(gold_text))
    
                predicted_tokens_clean = [
                    token for token in predicted_tokens
                    if token != predictor.sos and token != predictor.eos and token != predictor.blank
                ]
                predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]
    
                tokens_cpu = tokens.cpu().tolist() 
                gold_text = [predictor.tokenizer[token] for token in tokens_cpu[0] if token != predictor.blank]
                predicted_text_str = ' '.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                gold_text_str = ' '.join(gold_text)
                
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
