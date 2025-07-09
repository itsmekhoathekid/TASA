import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models import R_TASA_Transformer, CTCLoss, CrossEntropyLoss, add_nan_hook
from tqdm import tqdm
import argparse
import yaml
import os 
from models.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import logging
from utils import logg

# Cấu hình logger


def reload_model(model, optimizer, checkpoint_path):
    """
    Reload model and optimizer state from a checkpoint.
    """
    past_epoch = 0
    path_list = [path for path in os.listdir(checkpoint_path)]
    if len(path_list) > 0:
        for path in path_list:
            past_epoch = max(int(path.split("_")[-1]), past_epoch)
        
        load_path = os.path.join(checkpoint_path, f"transformer_transducer_epoch_{past_epoch}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No checkpoint found. Starting from scratch.")
    
    return past_epoch+1, model, optimizer


def train_one_epoch(model, dataloader, optimizer, criterion_ctc, criterion_ep, device, ctc_weight):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        speech = batch["fbank"].to(device)
        tokens_eos = batch["text"].to(device)
        speech_mask = batch["fbank_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        text_len = batch["text_len"].to(device)
        tokens = batch["tokens"].to(device)
        tokens_lens = batch["tokens_lens"].to(device)

        optimizer.zero_grad()

        enc_out , dec_out, enc_input_lengths   = model(
            src = speech, 
            tgt = decoder_input,
            src_mask = speech_mask,
            tgt_mask = text_mask
        )  # [B, T_text, vocab_size]
        
        # Bỏ <s> ở đầu nếu có
        loss = criterion_ctc(enc_out, tokens_eos, enc_input_lengths, text_len)
        # loss = criterion_ep(dec_out, tokens_eos)
        
        # print(f"Loss CTC: {loss_ctc.item()}, Loss EP: {loss_ep.item()}")
        # loss = loss_ctc * ctc_weight + loss_ep * (1- ctc_weight)
        loss.backward()

        # nan_grad = False
        # for name, param in model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"❌ NaN in gradient of: {name}")
        #         nan_grad = True

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # === In loss từng batch ===
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average training loss: {avg_loss:.4f}")
    return avg_loss


from torchaudio.functional import rnnt_loss

def evaluate(model, dataloader, optimizer, criterion_ctc, criterion_ep, device, ctc_weight):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            speech = batch["fbank"].to(device)
            tokens_eos = batch["text"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            text_len = batch["text_len"].to(device)
            tokens = batch["tokens"].to(device)
            tokens_lens = batch["tokens_lens"].to(device)

            optimizer.zero_grad()

            enc_out , dec_out, enc_input_lengths   = model(
                src = speech, 
                tgt = decoder_input,
                src_mask = speech_mask,
                tgt_mask = text_mask
            )  # [B, T_text, vocab_size]
            
            # Bỏ <s> ở đầu nếu có
            loss_ctc = criterion_ctc(enc_out, tokens, enc_input_lengths, tokens_lens)
            loss_ep = criterion_ep(dec_out, tokens_eos)
            
            loss = loss_ctc * ctc_weight + loss_ep * (1- ctc_weight)

            total_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average validation loss: {avg_loss:.4f}")
    return avg_loss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def main():
    from torch.optim import Adam
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']
    logg(training_cfg['logg'])

    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        json_path=training_cfg['train_path'],
        vocab_path=training_cfg['vocab_path'],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        collate_fn = speech_collate_fn
    )

    dev_dataset = Speech2Text(
        json_path=training_cfg['dev_path'],
        vocab_path=training_cfg['vocab_path']
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        collate_fn = speech_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = R_TASA_Transformer(
        in_features=config['model']['in_features'], 
        vocab_size=len(train_dataset.vocab),
        n_enc_layers=config['model']['n_enc_layers'],
        n_dec_layers=config['model']['n_dec_layers'],
        d_model=config['model']['d_model'],
        ff_size=config['model']['ff_size'],
        h=config['model']['h'],
        p_dropout=config['model']['p_dropout']
    ).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # add_nan_hook(model)  # Thêm hook để kiểm tra NaN trong model

    # === Khởi tạo loss ===
    # Giả sử <blank> = 0, và bạn chưa dùng reduction 'mean' toàn bộ batch
    criterion_ctc = CTCLoss(
        blank = train_dataset.vocab.get_blank_token(),
        reduction='mean',
        zero_infinity=True
    ).to(device)

    criterion_pe = CrossEntropyLoss(
        ignore_index=train_dataset.vocab.get_pad_token(),
        reduction='mean',

    ).to(device)

    ctc_weight = config['training']['ctc_weight']

    optimizer = Optimizer(model.parameters(), config['optim'])

    # ===Scheduler===
    scheduler = ReduceLROnPlateau(
        optimizer.optimizer,  # because you're using a wrapper class
        mode='min',
        factor=0.5,
        patience=2,
    )

    # === Huấn luyện ===

    start_epoch = 1
    if config['training']['reload']:
        checkpoint_path = config['training']['save_path']
        start_epoch, model, optimizer = reload_model(model, optimizer, checkpoint_path)
    num_epochs = config["training"]["epochs"]

    
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_ctc, criterion_pe, device, ctc_weight)
        val_loss = evaluate(model, dev_loader, optimizer, criterion_ctc, criterion_pe, device, ctc_weight)

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        # Save model checkpoint

        model_filename = os.path.join(
            config['training']['save_path'],
            f"R_TRANS_TASA_epoch_{epoch}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

        # Step scheduler with validation loss
        scheduler.step(val_loss)

        # Early stopping nếu lr quá nhỏ
        current_lr = optimizer.optimizer.param_groups[0]["lr"]
        if current_lr < 1e-6:
            logging.info('Learning rate quá thấp. Kết thúc training.')
            break


if __name__ == "__main__":
    main()

# 3