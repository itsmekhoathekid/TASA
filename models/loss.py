
# implement CTC loss

import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(self, blank = 0, reduction='mean', zero_infinity=False):
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
        
        log_probs = log_probs.transpose(1, 0) # B, T, C -> T, B, C
        
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
        return self.cross_entropy_loss(logits, targets)  # [B] -> [B, C]