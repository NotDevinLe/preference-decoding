# from utils import bernoulli_gumbel_soft, straight_through
import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Optional

class SparseMaskModel(nn.Module):
    def __init__(self, d, k, sparsity_weight=0.1):
        super().__init__()
        self.d = d
        self.k = k  
        self.sparsity_weight = sparsity_weight
        
        # Simple linear encoder/decoder (exactly like original gumbel.py)
        self.encoder = nn.Linear(d, k, bias=False)
        self.decoder = nn.Linear(k, d, bias=False)
        
        # Learnable mask parameters - logits for Gumbel-Softmax
        # Shape: [d] for shared feature masks across all components
        self.mask_logits = nn.Parameter(torch.zeros(d))

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
