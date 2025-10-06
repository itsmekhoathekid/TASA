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

# class GreedyMutiplePredictor:
#     def __init__(self, model, vocab, device, max_len=100, n_heads = 3, top_k = 1):
#         self.model = model
#         self.sos = vocab.get_sos_token()
#         self.eos = vocab.get_eos_token()
#         self.blank = vocab.get_blank_token()
#         self.tokenizer = vocab.itos
#         self.device = device
#         self.max_len = max_len
#         self.n_heads = n_heads
#         self.top_k = top_k

#     def greedy_decode(self, src, src_mask):
#         enc_out, src_mask = self.model.encode(src, src_mask)
#         decoder_input = torch.tensor([[self.sos]], dtype=torch.long).to(self.device)

#         for _ in range(self.max_len):
#             decoder_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)
#             dec_out = self.model.decode(decoder_input, enc_out, src_mask, decoder_mask)
#             list_prob = [head[:, -1, :]  for head in dec_out]
#             list_token = [torch.argmax(p, dim=-1).item() for p in list_prob]

#             list_word = [list_token[0]]
#             for i in range(self.n_heads-1):
#                 stack = decoder_input 
#                 prob = list_prob[i]
#                 # for prob in list_prob:
#                 _, token = torch.max(prob, dim=1)  # [B]
#                 token = torch.tensor([[token.item()]], dtype=torch.long).to(self.device)
#                 stack = torch.cat([stack, token], dim=1)
#                 decoder_mask = causal_mask(src.size(0), stack.size(1)).to(self.device)
#                 dec_out = self.model.decode(stack, enc_out, src_mask, decoder_mask)
#                 prob = dec_out[0][:, -1, :]  # [B, vocab_size]
#                 pred_token = torch.argmax(prob, dim=-1).item()
#                 # list_word.append(pred_token)
#                 if pred_token == list_token[i+1]:
#                     list_word.append(pred_token)
            
            
#             # if next_token not in [self.sos, self.eos, self.blank]:
#             next_token_tensor = torch.tensor([list_word], dtype=torch.long).to(self.device)
#             # print("Next token tensor : ", next_token_tensor.shape)
#             # print("Decoder input before cat : ", decoder_input.shape)
#             decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)
#             # print("Decoder input after cat : ", decoder_input.shape)

#             if self.eos in list_word:
#                 break
        
#         return decoder_input.squeeze(0).cpu().numpy()

import torch

class GreedyMutiplePredictor:
    def __init__(self, model, vocab, device, max_len=200, n_heads=4, tau=0.7):
        self.model = model
        self.sos = vocab.get_sos_token()
        self.eos = vocab.get_eos_token()
        self.blank = vocab.get_blank_token()
        self.tokenizer = vocab.itos
        self.device = device
        self.max_len = max_len
        self.n_heads = n_heads
        self.tau = tau  # threshold verify probability

    @torch.no_grad()
    def greedy_decode(self, src, src_mask):
        # ====== Encode ======
        enc_out, src_mask = self.model.encode(src, src_mask)
        decoder_input = torch.tensor([[self.sos]], dtype=torch.long).to(self.device)

        for step in range(self.max_len):
            # causal mask for current decoder input
            decoder_mask = causal_mask(src.size(0), decoder_input.size(1)).to(self.device)

            # ====== 1️⃣ Predict phase (multi-head prediction) ======
            dec_out_heads = self.model.decode(decoder_input, enc_out, src_mask, decoder_mask)
            probs_heads = [torch.softmax(h[:, -1, :], dim=-1) for h in dec_out_heads]

            # lấy top-1 token từ mỗi head
            predicted_tokens = [torch.argmax(p, dim=-1).item() for p in probs_heads]

            # nếu head 1 predict EOS thì dừng luôn
            if predicted_tokens[0] == self.eos:
                break

            # ====== 2️⃣ Build batched prefix for verify ======
            prefix_list = []
            stack = decoder_input.clone()
            for tok in predicted_tokens[:-1]:  # không cần token cuối
                tok_tensor = torch.tensor([[tok]], dtype=torch.long).to(self.device)
                stack = torch.cat([stack, tok_tensor], dim=1)
                prefix_list.append(stack.clone())

            # nếu không có prefix nào (chỉ còn 1 head) thì break
            if len(prefix_list) == 0:
                tok_tensor = torch.tensor([[predicted_tokens[0]]], dtype=torch.long).to(self.device)
                decoder_input = torch.cat([decoder_input, tok_tensor], dim=1)
                continue

            # pad các prefix thành batch
            max_len = max(p.size(1) for p in prefix_list)
            padded = []
            for p in prefix_list:
                pad_len = max_len - p.size(1)
                if pad_len > 0:
                    pad = torch.full((1, pad_len), self.blank, dtype=torch.long).to(self.device)
                    p = torch.cat([p, pad], dim=1)
                padded.append(p)
            prefix_stack = torch.cat(padded, dim=0)  # [K-1, T_max]

            # ====== 3️⃣ Verify phase (forward 1 batch qua P₁) ======
            src_mask_batch = src_mask.expand(prefix_stack.size(0), *src_mask.shape[1:])
            decoder_mask = causal_mask(src.size(0), prefix_stack.size(1)).to(self.device)
            dec_out_verify = self.model.decode(prefix_stack, enc_out, src_mask_batch, decoder_mask)

            # dùng P₁ để lấy xác suất cho token tiếp theo
            p1_logits = dec_out_verify[0][:, -1, :]  # [K-1, vocab]
            p1_probs = torch.softmax(p1_logits, dim=-1)

            # ====== 4️⃣ Accept tokens ======
            verified_tokens = []
            for i, tok in enumerate(predicted_tokens[1:]):  # bỏ head1
                prob = p1_probs[i, tok].item()
                if prob >= self.tau:
                    verified_tokens.append(tok)
                else:
                    break

            # luôn chấp nhận token đầu tiên (head1)
            verified_tokens.insert(0, predicted_tokens[0])

            # nối vào decoder_input
            new_tokens = torch.tensor([verified_tokens], dtype=torch.long).to(self.device)
            decoder_input = torch.cat([decoder_input, new_tokens], dim=1)

            # nếu có EOS trong verified tokens → dừng
            if self.eos in verified_tokens:
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

    predictor = GreedyMutiplePredictor(model, train_dataset.vocab, device)
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
    
                predicted_tokens_clean = [
                    token for token in predicted_tokens
                    if token != predictor.sos and token != predictor.eos and token != predictor.blank
                ]
                predicted_text = [predictor.tokenizer[token] for token in predicted_tokens_clean]
    
                tokens_cpu = tokens.cpu().tolist() 
                gold_text = [predictor.tokenizer[token] for token in tokens_cpu[0] if token != predictor.blank]
                gold_text_str = ' '.join(gold_text)

                predicted_text_str = ' '.join([t for t in predicted_text if t != predictor.blank and t != predictor.eos])

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
