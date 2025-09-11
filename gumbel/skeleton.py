from utils import bernoulli_gumbel_soft, straight_through
import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Optional

class SparseMaskModel(nn.Module):
    def __init__(self, d, k, sparsity_weight=1e-4):
        super().__init__()
        self.encoder = nn.Linear(d, k, bias=False)
        self.decoder = nn.Linear(k, d, bias=False)
        self.mask_logits = nn.Parameter(torch.zeros(d))
        self.sparsity_weight = sparsity_weight

    def forward_decode_hard_soft(self, X, ell, tau, gated_st=True):
        # sample soft/hard with current logits ell
        m_soft, m_hard = bernoulli_gumbel_soft(ell, tau)

        # HARD path (efficient): column slice by active attrs
        idx_on = torch.nonzero(m_hard, as_tuple=False).squeeze(1)
        if idx_on.numel() == 0:
            z_hard = torch.zeros(X.size(0), self.decoder.in_features, device=X.device)
        else:
            X_sel = X.index_select(1, idx_on)              # [N, s]
            W = self.encoder.weight                        # [k, d]
            W_sel = W.index_select(1, idx_on)              # [k, s]
            z_hard = X_sel @ W_sel.t()                     # [N, k]

        xhat_hard = self.decoder(z_hard)

        # SOFT path (dense; used only for backward)
        z_soft = (X * m_soft) @ self.encoder.weight.t()    # [N, k]
        xhat_soft = self.decoder(z_soft)

        # Combine (forward = hard, backward = soft)
        xhat = xhat_hard + (xhat_soft - xhat_hard).detach()

        # Optionally produce a *gated* proxy mask for logging or extra losses
        m = straight_through(m_soft, m_hard, gated=gated_st)
        return xhat, m, (m_soft, m_hard)
    
    def get_masks(self, training=True):
        """Get mask for input dimensions (soft during training for gradients)"""
        if training:
            # During training, use SOFT mask for gradients
            m_soft, m_hard = bernoulli_gumbel_soft(self.mask_logits, tau=1.0)
            # Use straight-through estimator: hard forward, soft backward
            return straight_through(m_soft, m_hard, gated=False)
        else:
            # During inference, use hard thresholding
            return (torch.sigmoid(self.mask_logits) > 0.5).float()
    
    def forward(self, x):
        """Forward pass with masked input features (like original gumbel.py)"""
        # Get masks
        masks = self.get_masks(training=self.training)  # [d]
        
        # Apply mask to input features
        x_masked = x * masks  # Broadcasting: [batch_size, d] * [d]
        
        # Encode masked input
        z = self.encoder(x_masked)
        
        # Decode back to full input space
        x_hat = self.decoder(z)
        
        return z, x_hat, masks
