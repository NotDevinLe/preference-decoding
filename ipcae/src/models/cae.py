import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.gumbel_distrib import (
    GumbelDistribution,
    GumbelDistributionBijectiveSoftmax,
    GumbelDistributionLogits,
    GumbelDistributionSoftmaxAct,
    GumbelDistributionSoftPlus,
)



class ConcreteLinear(pl.LightningModule):
    def __init__(
        self,
        input_dim=6,
        mask_ratio=None,
        k=None,
        dim_ip=0,
        pi_dropout=0,
        IP_dropout=0,
        pi_layer_norm=False,
        gumbel_learn_mode="logits",
        IP_initialization="random",
        IP_weights="shared",
        IP_bias=True,
        no_gumbel_noise=False,
        marginal_initialization="random",
    ):
        super().__init__()
        self.model_family = "cae"
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        self.no_gumbel_noise = no_gumbel_noise
        assert not (
            mask_ratio is not None and k is not None
        ), "Either specify mask ratio or k"
        if mask_ratio:
            self.k = int((1 - mask_ratio) * input_dim)
        else:
            self.k = k

        gumbel_args = dict(
            num_categories=input_dim,
            num_distributions=self.k,
            marginal_initialization=marginal_initialization,
            dim_ip=dim_ip,
            pi_dropout=pi_dropout,
            IP_dropout=IP_dropout,
            layer_norm=pi_layer_norm,
            IP_initialization=IP_initialization,
            IP_weights=IP_weights,
            IP_bias=IP_bias,
        )

        if gumbel_learn_mode == "pi":
            self.gumbel_distrib = GumbelDistribution(**gumbel_args)
        elif gumbel_learn_mode == "logits":
            self.gumbel_distrib = GumbelDistributionLogits(**gumbel_args)
        elif gumbel_learn_mode == "softmax":
            self.gumbel_distrib = GumbelDistributionSoftmaxAct(**gumbel_args)
        elif gumbel_learn_mode == "softplus":
            self.gumbel_distrib = GumbelDistributionSoftPlus(**gumbel_args)
        elif gumbel_learn_mode == "bij_softmax":
            self.gumbel_distrib = GumbelDistributionBijectiveSoftmax(**gumbel_args)
        else:
            raise NotImplementedError(
                f"Incorrect gumbel_mode'{gumbel_learn_mode}', choose between 'pi', 'softmax', 'logits' or 'bij_softmax'"
            )

        self.decoder = nn.Linear(self.k, input_dim, device=self.device)

    def feature_select(
        self, X, temperature, random, hard=False, eeg_threshold=None, rao_samples=0
    ):
        num_batches = X.shape[0]
        m, distrib_dict = self.gumbel_distrib.batch_sample_joint(
            num_batches,
            temperature,
            random,
            hard=hard,
            eeg_threshold=eeg_threshold,
            rao_samples=rao_samples,
            no_gumbel_noise=self.no_gumbel_noise,
        )
        u = torch.bmm(m, X.unsqueeze(-1))
        u = u.squeeze(-1)
        return u, distrib_dict, m.detach()

    def check_m_convergence(self, m, tol=0.99, mode="all"):
        assert mode in ["any", "all"]

        maxes, _ = torch.max(m, dim=2)
        means = torch.mean(maxes, dim=1)

        convg = (
            torch.any(means > tol).item()
            if mode == "any"
            else torch.all(means > tol).item()
        )
        converge_dict = {
            "convergence": convg,
            "mean_mean_max": torch.mean(means, dim=0).item(),
        }
        return converge_dict

    def forward(
        self, X, random, temperature, hard=False, eeg_threshold=None, rao_samples=0
    ):
        u, distrib_dict, m = self.feature_select(
            X,
            temperature,
            random,
            hard=hard,
            eeg_threshold=eeg_threshold,
            rao_samples=rao_samples,
        )
        X_rec = self.decoder(u)
        loss = F.mse_loss(X, X_rec, reduction="mean")
        frob_norm = torch.norm(X - X_rec, p="fro", dim=-1) / self.input_dim
        frob_norm = frob_norm.mean()
        converge_dict = self.check_m_convergence(m)
        returns = {
            "loss": loss,
            "frob_norm": frob_norm,
            "distrib_dict": distrib_dict,
            "converge_dict": converge_dict,
        }
        return returns

class ConcreteMLP(ConcreteLinear):
    def __init__(
        self,
        input_dim=77,
        decoder_hiddens=[],
        dropout=0.0,
        norm_layer=nn.LayerNorm,
        mask_ratio=None,
        k=None,
        dim_ip=0,
        pi_dropout=0,
        IP_dropout=0,
        pi_layer_norm=False,
        gumbel_learn_mode="logits",
        IP_initialization="random",
        IP_weights="shared",
        IP_bias=True,
        no_gumbel_noise=False,
        marginal_initialization="random",
    ):
        super().__init__(
            input_dim,
            mask_ratio,
            k,
            dim_ip,
            pi_dropout,
            IP_dropout,
            pi_layer_norm,
            gumbel_learn_mode,
            IP_initialization=IP_initialization,
            IP_weights=IP_weights,
            IP_bias=IP_bias,
            no_gumbel_noise=no_gumbel_noise,
            marginal_initialization=marginal_initialization,
        )

        decoder_hiddens = [self.k] + decoder_hiddens + [input_dim]
        nets_dec = []
        for i in range(len(decoder_hiddens) - 1):
            nets_dec += [nn.Linear(decoder_hiddens[i], decoder_hiddens[i + 1])]
            if i < len(decoder_hiddens) - 2:
                if norm_layer is not None:
                    nets_dec += [norm_layer(decoder_hiddens[i + 1])]
                nets_dec += [nn.LeakyReLU(0.2)]
                nets_dec += [nn.Dropout(dropout)]
        self.decoder = nn.Sequential(*nets_dec)

class ConcreteLinearRandom(ConcreteLinear):
    def __init__(
        self,
        *args,
        fixed_onehot: bool = True,
        without_replacement: bool = True,
        rng: Optional[torch.Generator] = None,
        **kwargs
    ):
        """
        fixed_onehot: if True, rows of m are one-hot; else use a random simplex distribution per row.
        without_replacement: only used when fixed_onehot=True. If K > D, this will be ignored.
        rng: optional torch.Generator for reproducibility.
        """
        super().__init__(*args, **kwargs)

        D = self.input_dim
        K = self.k

        if rng is None:
            rng = torch.Generator(device=self.device)

        if without_replacement and K <= D:
            indices = torch.randperm(D, generator=rng, device=self.device)[:K]
        else:
            indices = torch.randint(0, D, (K,), generator=rng, device=self.device)

        m_fixed = F.one_hot(indices, num_classes=D).to(torch.float32)  # [K, D]
        self.register_buffer("m_fixed", m_fixed)          # non-trainable, moves with device
        self.register_buffer("m_fixed_indices", indices)  # for easy logging
        self._m_mode = "fixed_onehot"

    def feature_select(
        self, X, temperature, random, hard=False, eeg_threshold=None, rao_samples=0
    ):
        """
        Ignore the gumbel distribution entirely and always use the fixed m.
        """
        B = X.shape[0]
        m = self.m_fixed.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        u = torch.bmm(m, X.unsqueeze(-1)).squeeze(-1)

        distrib_dict = {
            "num_categories": self.input_dim,
            "current_pi": self.m_fixed.mean(dim=0, keepdim=True),
            "mode": self._m_mode,
            "indices": None if (self.m_fixed_indices < 0).any() else self.m_fixed_indices.tolist(),
            "GJS": torch.tensor(0.0, device=self.device),
            "EEG": torch.tensor(0.0, device=self.device),
        }

        return u, distrib_dict, m.detach()


# MNIST
def cae_MLP_MNIST(**kwargs):  # reconstruction model
    return ConcreteMLP(input_dim=784, decoder_hiddens=[200], **kwargs)


def cae_MLP_attributes(**kwargs):  # reconstruction model
    return ConcreteMLP(input_dim=400, decoder_hiddens=[200], **kwargs)
