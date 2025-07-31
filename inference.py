import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models import R_TASA_Transformer
from tqdm import tqdm
import argparse
import yaml
import os 
from utils import logg, causal_mask


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, vocab_len: int, device: torch.device) -> R_TASA_Transformer:
    checkpoint_path = os.path.join(
        config['training']['save_path'],
        f"R_TRANS_TASA_epoch_85"
    )
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = R_TASA_Transformer(
        in_features=config['model']['in_features'], 
        vocab_size=vocab_len,
        n_enc_layers=config['model']['n_enc_layers'],
        n_dec_layers=config['model']['n_dec_layers'],
        d_model=config['model']['d_model'],
        ff_size=config['model']['ff_size'],
        h=config['model']['h'],
        p_dropout=config['model']['p_dropout']
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
            next_token_tensor = torch.tensor([[next_token.item()]], dtype=torch.long).to(self.device)
            decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

            if next_token == self.eos:
                break
        
        return decoder_input.squeeze(0).cpu().numpy()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Speech2Text(
        json_path=config['training']['train_path'],
        vocab_path=config['training']['vocab_path']
    )
    test_dataset = Speech2Text(
        json_path=config['training']['train_path'],
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

    model = load_model(config, vocab_len, device)

    predictor = GreedyPredictor(model, train_dataset.vocab, device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            src = batch['fbank'].to(device)
            src_mask = batch['fbank_mask'].to(device)
            tokens = batch["tokens"].to(device)
            # print("src shape : ", src.shape)
            # print("src mask : ", src_mask.shape)

            predicted_tokens = predictor.greedy_decode(src, src_mask)
            predicted_text = [predictor.tokenizer[token] for token in predicted_tokens]
            print("Predicted text: ", ' '.join(predicted_text))
            tokens_cpu = tokens.cpu().tolist() 
            gold_text = [predictor.tokenizer[token] for token in tokens_cpu[0]]
            print("Gold Text: ", ' '.join(gold_text))

if __name__ == "__main__":
    main()
