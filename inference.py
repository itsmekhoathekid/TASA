import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models import Transformer
from tqdm import tqdm
import argparse
import yaml
import os 
from utils import logg, causal_mask, calculate_mask
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
            prob = dec_out[0][:, -1, :]  # [B, vocab_size]

            _, next_token = torch.max(prob, dim=1)  # [B]

            if next_token not in [self.sos, self.eos, self.blank]:
                next_token_tensor = torch.tensor([[next_token.item()]], dtype=torch.long).to(self.device)
                decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

            if next_token == self.eos:
                break
        
        return decoder_input.squeeze(0).cpu().numpy()

class GreedyMPStackPredictor:
    def __init__(self, model, vocab, device, max_len=100):
        self.model = model
        self.sos = vocab.get_sos_token()
        self.eos = vocab.get_eos_token()
        self.blank = vocab.get_blank_token()
        self.space = vocab.get_space_token()
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
    def greedy_decode(self, src, src_mask):
        enc_out, src_mask = self.model.encode(src, src_mask)
        decoder_input = torch.tensor([[[self.sos, self.sos, self.sos]]], dtype=torch.long).to(self.device)
        
        for _ in range(self.max_len):
            decoder_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)
            # print("decoder mask : ", decoder_mask.shape)
            # print("enc out shape : ", enc_out.shape)
            dec_out = self.model.decode(decoder_input, enc_out, src_mask, decoder_mask)
            initial = dec_out[0][:, -1, :]  # [B, vocab_size]
            rhyme = dec_out[1][:, -1, :]  # [B, vocab_size]
            tone = dec_out[2][:, -1, :]  # [B, vocab_size]

            _, initial_tokens =  torch.max(initial, dim=1)  # [B]
            _, rhyme_tokens =  torch.max(rhyme, dim=1)  # [B]
            _, tone_tokens =  torch.max(tone, dim=1)  # [

            # if next_token not in [self.sos, self.eos, self.blank]:
            next_token_tensor = torch.tensor([[[initial_tokens.item(), rhyme_tokens.item(), tone_tokens.item()]]], dtype=torch.long).to(self.device)
            decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

            if self.eos in [initial_tokens, rhyme_tokens, tone_tokens]:
                break
        
        return decoder_input.squeeze(0).cpu().numpy()

import torch

import torch

class GreedyMutiplePredictor:
    def __init__(self, model, vocab, device, max_len=50, n_heads=3, tau=0.85):
        self.model = model
        self.sos = vocab.get_sos_token()
        self.eos = vocab.get_eos_token()
        self.blank = vocab.get_blank_token()
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
        self.n_heads = n_heads
        self.tau = tau  # threshold for verification

    @torch.no_grad()
    def greedy_decode(self, src, src_mask):
        # ===== 1. Encode =====
        enc_out, src_mask = self.model.encode(src, src_mask)
        decoder_input = torch.tensor([[self.sos]], dtype=torch.long, device=self.device)

        fwd_pred, fwd_verify = 0, 0

        for _ in range(self.max_len):
            # causal mask for current prefix
            decoder_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)

            # ===== 2. Predict phase (multi-head) =====
            dec_out_heads = self.model.decode(decoder_input, enc_out, src_mask, decoder_mask)
            fwd_pred += 1
            probs_heads = [torch.softmax(h[:, -1, :], dim=-1) for h in dec_out_heads]
            draft_tokens = [torch.argmax(p[0], dim=-1).item() for p in probs_heads]

            if draft_tokens[0] == self.eos:
                break

            # ===== 3. Verify phase (single forward, blockwise) =====
            # prefix + first (K−1) draft tokens
            seq_full = torch.cat(
                [decoder_input, torch.tensor([draft_tokens[:-1]], device=self.device)], dim=1
            )
            verify_mask = causal_mask(src.size(0), seq_full.size(1)).to(self.device)
            dec_out_verify = self.model.decode(seq_full, enc_out, src_mask, verify_mask)
            fwd_verify += 1

            # logits from main head p₁ for all time steps of seq_full
            p1_logits_all = dec_out_verify[0]            # [1, L+(K−1), V]
            L = decoder_input.size(1)
            p1_block = p1_logits_all[:, L-1 : L-1+self.n_heads, :]  # [1,K,V]
            

            # ===== 4. Find largest k̂ =====
            if self.tau is not None:
                p1_probs = torch.exp(p1_block[0])  # convert log-prob to prob
                ok = p1_probs[torch.arange(self.n_heads), torch.tensor(draft_tokens, device=self.device)] >= self.tau
                if ok.any():
                    false_idx = (~ok).nonzero(as_tuple=False)
                    k_hat = false_idx[0,0].item() if false_idx.numel() > 0 else self.n_heads
                else:
                    k_hat = 0
            else:
                p1_probs = torch.softmax(p1_block, dim=-1)[0]          # [K,V]
                p1_pred = torch.argmax(p1_probs, dim=-1)
                matches = (p1_pred == torch.tensor(draft_tokens, device=self.device))
                if matches.any():
                    false_idx = (~matches).nonzero(as_tuple=False)
                    k_hat = false_idx[0,0].item() if false_idx.numel() > 0 else self.n_heads
                else:
                    k_hat = 0

            # always accept head-1’s token
            if k_hat == 0:
                k_hat = 1
            accepted = draft_tokens[:k_hat]

            # ===== 5. Append accepted tokens =====
            decoder_input = torch.cat(
                [decoder_input, torch.tensor([accepted], device=self.device)], dim=1
            )

            if self.eos in accepted:
                break

        total_fw = fwd_pred + fwd_verify
        print(f"🧮 Total forward calls: {total_fw}")
        print(f"   ├─ Predict phase : {fwd_pred}")
        print(f"   └─ Verify phase  : {fwd_verify}")
        print(f"📏 Output length   : {decoder_input.size(1)}")
        print(f"⚡ Avg forward/token: {round(total_fw / decoder_input.size(1), 3)}")
        
        return decoder_input.squeeze(0).cpu().numpy()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch to load the model from")
    parser.add_argument("--type_decode", type=str, default="ar", help="Type of decoding: autoregressive or mtp") 
    parser.add_argument("--tau", type=float, default=None, help="Threshold for verification in MTP decoding")
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

    if args.type_decode == "mtp_stack":
        predictor = GreedyMPStackPredictor(model, train_dataset.vocab, device)
    elif args.type_decode == "mtp":
        predictor = GreedyMutiplePredictor(model, train_dataset.vocab, device, tau=args.tau)
    else:
        predictor = GreedyPredictor(model, train_dataset.vocab, device)
    # predictor = GreedyMutiplePredictor(model, train_dataset.vocab, device, tau=args.tau) if args.type_decode == "mtp" else GreedyPredictor(model, train_dataset.vocab, device)
    # predictor = GreedyPredictor(model, train_dataset.vocab, device)
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

                if args.type_decode != "mtp_stack":
                    predicted_tokens_clean = [
                        token for token in predicted_tokens
                        if token != predictor.sos and token != predictor.eos and token != predictor.blank
                    ]
                    predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]

                    tokens_cpu = tokens.cpu().tolist() 
                    gold_text = [predictor.tokenizer[token] for token in tokens_cpu[0] if token != predictor.blank]
                    gold_text_str = ' '.join(gold_text)

                    predicted_text_str = ' '.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                else:
                    # Chuyển từ dạng [[i1, r1, t1], [i2, r2, t2], ...] sang [i1, r1, t1, i2, r2, t2, ...]
                    predicted_tokens_flat = []
                    for triplet in predicted_tokens:
                        predicted_tokens_flat.extend(triplet)
                        predicted_tokens_flat += [predictor.space]
                    
                    predicted_tokens_clean = [
                        token for token in predicted_tokens_flat
                        if token != predictor.sos and token != predictor.eos and token != predictor.blank
                    ]
                    predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]
    
                    tokens_cpu_flat = []
                    tokens_cpu = tokens.cpu().tolist() 
                    for triplet in tokens_cpu[0]:
                        tokens_cpu_flat.extend(triplet)
                        tokens_cpu_flat += [predictor.space]
                    # print(tokens_cpu_flat)
                    gold_text = [predictor.tokenizer[token] for token in tokens_cpu_flat if token != predictor.blank]
                    gold_text_str = ''.join(gold_text)
                    gold_text_str = gold_text_str.replace(predictor.tokenizer[predictor.space], ' ')
                    predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                    predicted_text_str = predicted_text_str.replace(predictor.tokenizer[predictor.space], ' ')

                if config['training']['type'] == "phoneme":
                    predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                    space_token = vocab.get("<space>")
                    predicted_text_str = predicted_text_str.replace(predictor.tokenizer[space_token], ' ')

                    gold_text = ''.join([predictor.tokenizer[token] for token in tokens_cpu[0] if token != predictor.blank])
                    gold_text_str = gold_text.replace(predictor.tokenizer[space_token], ' ')
                elif config['training']['type'] == "char":
                    predicted_text_str = ''.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])
                    gold_text_str = ''.join([predictor.tokenizer[token] for token in tokens_cpu[0] if token != predictor.blank])
                
                
                
                
                all_gold_texts.append(gold_text_str)
                all_predicted_texts.append(predicted_text_str)
                print("Predicted text: ", predicted_text_str)
                print("Gold Text: ", gold_text_str)

                wer_score = wer(gold_text_str, predicted_text_str)
                cer_score = cer(gold_text_str, predicted_text_str)
                print(f"WER: {wer_score:.4f}, CER: {cer_score:.4f}")
                # raise
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
