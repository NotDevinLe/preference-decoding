import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Optional

# ---- Gumbel helpers ----
def sample_gumbel(shape, device):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + 1e-20) + 1e-20)

def bernoulli_gumbel_soft(logits, tau):
    g = sample_gumbel(logits.shape, logits.device)
    m_soft = torch.sigmoid((logits + g) / tau)
    m_hard = (m_soft > 0.5).float()
    return m_soft, m_hard

# vanilla ST: grads everywhere; gated ST: grads only where hard==1
def straight_through(m_soft, m_hard, gated=False):
    if gated:
        return m_hard + (m_soft - m_hard) * m_hard.detach()
    else:
        return (m_hard - m_soft).detach() + m_soft

# reward scoring now handled directly in collector_server.py
