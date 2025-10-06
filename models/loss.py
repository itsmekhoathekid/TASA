
# implement CTC loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.losses import kldiv_loss

class CTCLoss(nn.Module):
    def __init__(self, blank = 4, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.blank_index = blank
        self.reduction = reduction
    
    def forward(self, log_probs, targets, input_lens, target_lens):
        """
        Args:
            log_probs (Tensor): Log probabilities of shape (T, N, C) where T is the input length,
                                N is the batch size, and C is the number of classes.
            targets (Tensor): Target indices of shape (N, S) where S is the maximum target length.
            input_lengths (Tensor): Lengths of each input sequence in the batch of shape (N,).
            target_lengths (Tensor): Lengths of each target sequence in the batch of shape (N,).
        """
        log_probs = log_probs.transpose(0, 1)

        if self.reduction == "batchmean":
            reduction_loss = "sum"
        elif self.reduction == "batch":
            reduction_loss = "none"
        else:
            reduction_loss = self.reduction
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lens,
            target_lens,
            self.blank_index,
            zero_infinity=True,
            reduction=reduction_loss,
        )

        if self.reduction == "batchmean":
            return loss / targets.shape[0]
        elif self.reduction == "batch":
            N = loss.size(0)
            return loss.view(N, -1).sum(1) / target_lens.view(N, -1).sum(1)
        else:
            return loss

class Kldiv_Loss(nn.Module):
    def __init__(self, reduction='mean', pad_idx=None):
        super(Kldiv_Loss, self).__init__()
        self.pad_idx = pad_idx
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Logits of shape (N, C) where N is the batch size and C is the number of classes.
            targets (Tensor): Target indices of shape (N,).
        """
        logits = logits.log_softmax(dim=-1)
        
        return kldiv_loss(log_probabilities= logits, targets=targets, pad_idx = self.pad_idx, reduction = self.reduction)

class CELoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='mean'):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Logits shape (N, C)/(N,T,C)
        """
        if logits.dim() == 3:
            logits = logits.transpose(1,2)
 
        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss
