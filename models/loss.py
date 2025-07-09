
# implement CTC loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCLoss(nn.Module):
    def __init__(self, blank = 4, reduction='mean', zero_infinity=False):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs (Tensor): Log probabilities of shape (T, N, C) where T is the input length,
                                N is the batch size, and C is the number of classes.
            targets (Tensor): Target indices of shape (N, S) where S is the maximum target length.
            input_lengths (Tensor): Lengths of each input sequence in the batch of shape (N,).
            target_lengths (Tensor): Lengths of each target sequence in the batch of shape (N,).
        """
        log_probs = F.log_softmax(log_probs, dim= - 1)  # Convert logits to log probabilities
        log_probs = log_probs.transpose(1, 0) # B, T, C -> T, B, C
        # assert not torch.isnan(log_probs).any(), "NaN in log_probs"
        # assert not torch.isnan(targets).any(), "NaN in targets"
        # assert (input_lengths >= target_lengths).all(), "input_lengths must >= target_lengths"
        
        # print(f"log_probs shape: {log_probs.shape}")
        # print(f"targets shape: {targets.shape}")
        # print(f"input_lengths shape: {input_lengths.shape}")
        # print(f"target_lengths shape: {target_lengths.shape}")
        
        # print("input lengths:", input_lengths)
        # print("targets:", targets)
        # print("target lengths:", target_lengths)
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=None):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Logits of shape (N, C) where N is the batch size and C is the number of classes.
            targets (Tensor): Target indices of shape (N,).
        """
        logits = logits.view(-1, logits.size(-1))    # [B*T, V]
        targets = targets.view(-1)

        assert not torch.isnan(logits).any(), "NaN in log_probs"
        assert not torch.isnan(targets).any(), "NaN in targets"

        return self.cross_entropy_loss(logits, targets)  # [B] -> [B, C]