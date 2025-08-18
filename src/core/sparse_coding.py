"""
Joint Sparse Coding with Global (Row) and Local (Elementwise) Sparsity
Implementation of Algorithm 1 from the paper.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm


class SparseCoding:
    """
    Joint sparse coding for attribute/persona selection.
    
    Solves: min_{B,W} ||Y - BW||_F^2 + λ₁||W||₁ + λ₂,₁||W||₂,₁ + β||B||_F^2
    
    Where:
        Y ∈ R^(dxU): Reward matrix (d datapoints, U personas)
        B ∈ R^(dxk): Basis matrix (d datapoints, k atoms)
        W ∈ R^(kxU): Sparse codes (k atoms, U personas)
    """
    
    def __init__(
        self,
        k: int,
        lambda1: float = 0.1,
        lambda21: float = 0.1,
        beta: float = 0.01,
        epsilon: float = 1e-4,
        max_iter: int = 1000,
        device: str = "cuda"
    ):
        """
        Initialize sparse coding solver.
        
        Args:
            k: Number of basis atoms (reduced dimension)
            lambda1: L1 regularization weight (elementwise sparsity)
            lambda21: L2,1 regularization weight (row sparsity)
            beta: Ridge regularization for basis
            epsilon: Pruning threshold for global sparsity
            max_iter: Maximum outer iterations
            device: Device for computation
        """
        self.k = k
        self.lambda1 = lambda1
        self.lambda21 = lambda21
        self.beta = beta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.device = device
        
        self.B = None  # Basis matrix
        self.W = None  # Sparse codes
        self.active_atoms = None  # Track which atoms are active
    
    def _initialize_basis(self, Y: torch.Tensor) -> torch.Tensor:
        """Initialize basis B with normalized random values."""
        d, U = Y.shape
        B = torch.randn(d, self.k, device=self.device)
        B = B / torch.norm(B, dim=0, keepdim=True)  # Column normalization
        return B
    
    def _compute_stepsize(self, B: torch.Tensor) -> float:
        """Compute step size η = (2||B^T B||₂)^(-1)."""
        BTB = B.T @ B
        # Compute spectral norm (largest singular value)
        spectral_norm = torch.linalg.norm(BTB, ord=2)
        return 1.0 / (2.0 * spectral_norm)
    
    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft thresholding operator: max(|x| - threshold, 0) * sign(x)."""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
    
    def _row_l21_prox(self, Z: torch.Tensor, eta: float) -> torch.Tensor:
        """
        Proximal operator for row-wise L2,1 norm.
        For each row j: W_j = max(0, 1 - η*λ₂,₁/(s+1e-12)) * Z_j
        where s = ||Z_j||₂
        """
        W = torch.zeros_like(Z)
        
        for j in range(Z.shape[0]):
            row_norm = torch.norm(Z[j, :], p=2)
            if row_norm > 1e-12:
                scale = torch.maximum(
                    torch.tensor(0.0, device=self.device),
                    1.0 - eta * self.lambda21 / (row_norm + 1e-12)
                )
                W[j, :] = scale * Z[j, :]
        
        return W
    
    def _code_step(self, Y: torch.Tensor, B: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Update codes W using proximal gradient method.
        """
        # Compute gradient: ∇f(W) = 2B^T(BW - Y)
        gradient = 2 * B.T @ (B @ W - Y)
        
        # Compute step size
        eta = self._compute_stepsize(B)
        
        # Gradient step
        R = W - eta * gradient
        
        # Positive soft-thresholding for L1
        Z = self._soft_threshold(R, eta * self.lambda1)
        Z = torch.relu(Z)  # Ensure non-negative
        
        # Row-wise L2,1 proximal operator
        W_new = self._row_l21_prox(Z, eta)
        
        return W_new
    
    def _basis_step(self, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Update basis B using ridge regression.
        B = YW^T(WW^T + βI)^(-1)
        """
        k_current = W.shape[0]
        
        # Ridge regression solution
        WWT = W @ W.T
        regularized = WWT + self.beta * torch.eye(k_current, device=self.device)
        
        # Solve using Cholesky decomposition for numerical stability
        try:
            L = torch.linalg.cholesky(regularized)
            # Solve L L^T X = W Y^T for X
            # First solve L Z = W Y^T for Z
            Z = torch.cholesky_solve((W @ Y.T).T, L)
            B_new = Y @ Z.T
        except:
            # Fallback to direct inverse if Cholesky fails
            B_new = Y @ W.T @ torch.linalg.inv(regularized)
        
        # Project to non-negative orthant
        B_new = torch.relu(B_new)
        
        # Column normalization
        col_norms = torch.norm(B_new, dim=0, keepdim=True)
        B_new = B_new / torch.maximum(col_norms, torch.tensor(1.0, device=self.device))
        
        return B_new
    
    def _global_prune(
        self,
        B: torch.Tensor,
        W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Remove inactive atoms based on row norms of W.
        """
        row_norms = torch.norm(W, dim=1, p=2)
        active_mask = row_norms > self.epsilon
        
        if torch.sum(active_mask) == 0:
            # Keep at least one atom
            max_idx = torch.argmax(row_norms)
            active_mask[max_idx] = True
        
        # Prune B and W
        B_pruned = B[:, active_mask]
        W_pruned = W[active_mask, :]
        
        return B_pruned, W_pruned, active_mask
    
    def fit(
        self,
        Y: torch.Tensor,
        verbose: bool = True
    ) -> Dict:
        """
        Fit sparse coding model to reward matrix Y.
        
        Args:
            Y: Reward matrix (d x U)
            verbose: Print progress information
            
        Returns:
            Dictionary with results and statistics
        """
        Y = Y.to(self.device)
        d, U = Y.shape
        
        # Initialize
        self.B = self._initialize_basis(Y)
        self.W = torch.zeros(self.k, U, device=self.device)
        
        history = {
            "reconstruction_error": [],
            "sparsity": [],
            "num_active_atoms": []
        }
        
        # Main optimization loop
        pbar = tqdm(range(self.max_iter), desc="Sparse Coding") if verbose else range(self.max_iter)
        
        for t in pbar:
            # Code step: Update W
            self.W = self._code_step(Y, self.B, self.W)
            
            # Basis step: Update B
            self.B = self._basis_step(Y, self.W)
            
            # Global pruning every 10 iterations
            if t > 0 and t % 10 == 0:
                self.B, self.W, self.active_atoms = self._global_prune(self.B, self.W)
                self.k = self.B.shape[1]  # Update k
            
            # Compute metrics
            reconstruction = self.B @ self.W
            error = torch.norm(Y - reconstruction, p='fro').item()
            sparsity = (torch.sum(self.W == 0).item()) / self.W.numel()
            num_active = self.W.shape[0]
            
            history["reconstruction_error"].append(error)
            history["sparsity"].append(sparsity)
            history["num_active_atoms"].append(num_active)
            
            if verbose and t % 10 == 0:
                pbar.set_postfix({
                    "error": f"{error:.4f}",
                    "sparsity": f"{sparsity:.3f}",
                    "atoms": num_active
                })
            
            # Early stopping if converged
            if t > 100 and len(history["reconstruction_error"]) > 10:
                recent_errors = history["reconstruction_error"][-10:]
                if np.std(recent_errors) / np.mean(recent_errors) < 1e-4:
                    if verbose:
                        print(f"Converged at iteration {t}")
                    break
        
        # Final statistics
        results = {
            "B": self.B.cpu(),
            "W": self.W.cpu(),
            "history": history,
            "final_k": self.k,
            "final_error": history["reconstruction_error"][-1],
            "final_sparsity": history["sparsity"][-1]
        }
        
        return results
    
    def get_selected_personas(self, threshold: float = 0.1) -> torch.Tensor:
        """
        Get indices of selected personas based on code weights.
        
        Args:
            threshold: Minimum weight threshold for selection
            
        Returns:
            Indices of selected personas
        """
        if self.W is None:
            raise ValueError("Model not fitted yet")
        
        # Sum absolute weights across atoms for each persona
        persona_weights = torch.sum(torch.abs(self.W), dim=0)
        
        # Select personas with weight above threshold
        selected = torch.where(persona_weights > threshold)[0]
        
        return selected
    
    def reconstruct(self) -> torch.Tensor:
        """Reconstruct the reward matrix using learned B and W."""
        if self.B is None or self.W is None:
            raise ValueError("Model not fitted yet")
        
        return self.B @ self.W
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            "B": self.B.cpu() if self.B is not None else None,
            "W": self.W.cpu() if self.W is not None else None,
            "k": self.k,
            "lambda1": self.lambda1,
            "lambda21": self.lambda21,
            "beta": self.beta,
            "epsilon": self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.B = checkpoint["B"].to(self.device) if checkpoint["B"] is not None else None
        self.W = checkpoint["W"].to(self.device) if checkpoint["W"] is not None else None
        self.k = checkpoint["k"]
        self.lambda1 = checkpoint["lambda1"]
        self.lambda21 = checkpoint["lambda21"]
        self.beta = checkpoint["beta"]
        self.epsilon = checkpoint["epsilon"]