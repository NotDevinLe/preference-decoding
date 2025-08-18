"""
Matrix Factorization with Group Sparsity for Persona Attribute Discovery
Solves: min_{B,W} 1/2 ||Y - BW||_F^2 + λ Σ_j ||W_j,:||_2 + β/2 ||B||_F^2

Given fixed dataset of 100 personas × 100 questions with precomputed rewards,
learns sparse global attributes that explain persona behavior.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
from tqdm import tqdm


class SparseCoding:
    """
    Matrix factorization with group sparsity for persona attribute discovery.
    
    Solves: min_{B,W} 1/2 ||Y - BW||_F^2 + λ Σ_j ||W_j,:||_2 + β/2 ||B||_F^2
    
    Where:
        Y ∈ R^(dxU): Reward matrix (d questions, U personas)
        B ∈ R^(dxk): Global attributes (d questions, k attributes)
        W ∈ R^(kxU): Persona loadings (k attributes, U personas)
        
    Group sparsity (L2,1) on W encourages attribute pruning.
    Ridge penalty on B provides stability.
    """
    
    def __init__(
        self,
        k: int,
        lmbda: float = 0.1,
        beta: float = 0.01,
        epsilon: float = 1e-3,
        max_iter: int = 1000,
        tol: float = 1e-6,
        init_method: str = "svd",
        normalize_rows: bool = True,
        normalize_cols: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize matrix factorization solver.
        
        Args:
            k: Number of attributes (reduced dimension)
            lmbda: L2,1 regularization weight (group sparsity)
            beta: Ridge regularization for B
            epsilon: Pruning threshold for attributes
            max_iter: Maximum iterations
            tol: Convergence tolerance
            init_method: Initialization method ("svd" or "random")
            normalize_rows: Whether to normalize rows of Y
            normalize_cols: Whether to normalize columns of B
            device: Device for computation
        """
        self.k_init = k
        self.k = k
        self.lmbda = lmbda
        self.beta = beta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.normalize_rows = normalize_rows
        self.normalize_cols = normalize_cols
        self.device = device
        
        self.B = None  # Global attributes
        self.W = None  # Persona loadings
        self.Y_normalized = None  # Preprocessed Y
        self.row_stats = None  # For denormalization
        self.active_mask = None  # Track active attributes
    
    def _preprocess_data(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Preprocess reward matrix with optional row normalization.
        
        Args:
            Y: Raw reward matrix (d x U)
            
        Returns:
            Preprocessed matrix
        """
        if not self.normalize_rows:
            self.row_stats = None
            return Y
        
        # Z-score normalization per row (question)
        row_means = torch.mean(Y, dim=1, keepdim=True)
        row_stds = torch.std(Y, dim=1, keepdim=True)
        row_stds = torch.maximum(row_stds, torch.tensor(1e-8, device=self.device))
        
        Y_normalized = (Y - row_means) / row_stds
        
        # Store stats for potential denormalization
        self.row_stats = {
            "means": row_means.cpu(),
            "stds": row_stds.cpu()
        }
        
        return Y_normalized
    
    def _initialize_basis(self, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize B and W using SVD or random initialization.
        
        Args:
            Y: Preprocessed reward matrix (d x U)
            
        Returns:
            Tuple of (B, W) matrices
        """
        d, U = Y.shape
        
        if self.init_method == "svd":
            # SVD initialization: Y ≈ U S V^T, take top k components
            U_svd, S, Vt = torch.linalg.svd(Y, full_matrices=False)
            k_effective = min(self.k, len(S))
            
            # B = U[:, :k] * sqrt(S[:k])
            B = U_svd[:, :k_effective] * torch.sqrt(S[:k_effective]).unsqueeze(0)
            
            # W = sqrt(S[:k]) * V[:k, :]
            W = torch.sqrt(S[:k_effective]).unsqueeze(1) * Vt[:k_effective, :]
            
            # Pad with zeros if k > rank
            if k_effective < self.k:
                B_pad = torch.zeros(d, self.k - k_effective, device=self.device)
                W_pad = torch.zeros(self.k - k_effective, U, device=self.device)
                B = torch.cat([B, B_pad], dim=1)
                W = torch.cat([W, W_pad], dim=0)
                
        else:  # random initialization
            B = torch.randn(d, self.k, device=self.device) * 0.1
            W = torch.randn(self.k, U, device=self.device) * 0.1
        
        # Ensure non-negativity and normalize columns of B
        B = torch.relu(B)
        if self.normalize_cols:
            col_norms = torch.norm(B, dim=0, keepdim=True)
            B = B / torch.maximum(col_norms, torch.tensor(1e-8, device=self.device))
        
        return B, W
    
    def _compute_stepsize(self, B: torch.Tensor, backtrack: bool = True) -> float:
        """
        Compute step size with optional backtracking.
        
        Args:
            B: Current basis matrix
            backtrack: Whether to use backtracking line search
            
        Returns:
            Step size
        """
        BTB = B.T @ B
        spectral_norm = torch.linalg.norm(BTB, ord=2)
        base_stepsize = 1.0 / (2.0 * spectral_norm + 1e-8)
        
        if backtrack:
            # Conservative step size
            return 0.5 * base_stepsize
        else:
            return base_stepsize
    
    
    def _group_soft_threshold(self, Z: torch.Tensor, eta: float) -> torch.Tensor:
        """
        Proximal operator for group (L2,1) penalty.
        For each row j: W_j = max(0, 1 - η*λ/||Z_j||₂) * Z_j
        
        Args:
            Z: Input matrix
            eta: Step size
            
        Returns:
            Thresholded matrix
        """
        W = torch.zeros_like(Z)
        threshold = eta * self.lmbda
        
        # Vectorized row-wise soft thresholding
        row_norms = torch.norm(Z, dim=1, p=2, keepdim=True)
        scale_factors = torch.maximum(
            torch.zeros_like(row_norms),
            1.0 - threshold / torch.maximum(row_norms, torch.tensor(1e-12, device=self.device))
        )
        W = scale_factors * Z
        
        return W
    
    def _code_step(self, Y: torch.Tensor, B: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Update persona loadings W using proximal gradient descent.
        Minimizes: 1/2 ||Y - BW||_F^2 + λ Σ_j ||W_j,:||_2
        """
        # Compute gradient: ∇f(W) = B^T(BW - Y)
        residual = B @ W - Y
        gradient = B.T @ residual
        
        # Compute step size
        eta = self._compute_stepsize(B)
        
        # Gradient step
        Z = W - eta * gradient
        
        # Apply group soft thresholding
        W_new = self._group_soft_threshold(Z, eta)
        
        return W_new
    
    def _basis_step(self, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Update global attributes B using ridge regression.
        Solves: min_B 1/2 ||Y - BW||_F^2 + β/2 ||B||_F^2
        Solution: B = YW^T(WW^T + βI)^(-1)
        """
        k_current = W.shape[0]
        
        # Ridge regression with improved numerical stability
        WWT = W @ W.T
        regularized = WWT + self.beta * torch.eye(k_current, device=self.device)
        
        # Add small jitter for numerical stability
        regularized += 1e-8 * torch.eye(k_current, device=self.device)
        
        try:
            # Preferred: Cholesky decomposition
            L = torch.linalg.cholesky(regularized)
            rhs = Y @ W.T
            B_new = torch.cholesky_solve(rhs.T, L).T
        except RuntimeError:
            try:
                # Fallback: LU decomposition
                B_new = Y @ W.T @ torch.linalg.inv(regularized)
            except RuntimeError:
                # Last resort: pseudoinverse
                B_new = Y @ W.T @ torch.linalg.pinv(regularized)
        
        # Optional: ensure non-negativity
        # B_new = torch.relu(B_new)
        
        # Optional: column normalization
        if self.normalize_cols:
            col_norms = torch.norm(B_new, dim=0, keepdim=True)
            B_new = B_new / torch.maximum(col_norms, torch.tensor(1e-8, device=self.device))
            
            # Rescale W to maintain reconstruction
            if hasattr(self, 'W') and self.W is not None:
                W = W * col_norms.T
        
        return B_new
    
    def _prune_attributes(
        self,
        B: torch.Tensor,
        W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Remove inactive attributes based on row norms of W.
        
        Args:
            B: Current basis matrix
            W: Current loading matrix
            
        Returns:
            Tuple of (B_pruned, W_pruned, active_mask)
        """
        row_norms = torch.norm(W, dim=1, p=2)
        
        # Dynamic threshold based on median
        if len(row_norms) > 1:
            median_norm = torch.median(row_norms)
            threshold = max(self.epsilon, 0.01 * median_norm.item())
        else:
            threshold = self.epsilon
            
        active_mask = row_norms > threshold
        
        # Keep at least one attribute
        if torch.sum(active_mask) == 0:
            max_idx = torch.argmax(row_norms)
            active_mask[max_idx] = True
        
        # Prune matrices
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
        
        # Preprocess data
        self.Y_normalized = self._preprocess_data(Y)
        
        # Initialize B and W
        self.B, self.W = self._initialize_basis(self.Y_normalized)
        
        history = {
            "reconstruction_error": [],
            "relative_error": [],
            "num_active_attributes": [],
            "group_sparsity": [],
            "frobenius_norm_B": [],
            "frobenius_norm_W": []
        }
        
        # Compute initial error
        Y_norm_sq = torch.norm(self.Y_normalized, p='fro').item() ** 2
        
        # Main optimization loop
        pbar = tqdm(range(self.max_iter), desc="Matrix Factorization") if verbose else range(self.max_iter)
        
        for t in pbar:
            # Store previous error for convergence check
            prev_error = history["reconstruction_error"][-1] if history["reconstruction_error"] else float('inf')
            
            # W-step: Update persona loadings
            self.W = self._code_step(self.Y_normalized, self.B, self.W)
            
            # B-step: Update global attributes  
            self.B = self._basis_step(self.Y_normalized, self.W)
            
            # Pruning every 10 iterations after warmup
            if t >= 20 and t % 10 == 0:
                self.B, self.W, self.active_mask = self._prune_attributes(self.B, self.W)
                self.k = self.B.shape[1]  # Update k
            
            # Compute metrics
            reconstruction = self.B @ self.W
            error = torch.norm(self.Y_normalized - reconstruction, p='fro').item()
            relative_error = error / np.sqrt(Y_norm_sq) if Y_norm_sq > 0 else error
            
            # Group sparsity: fraction of zero rows in W
            row_norms = torch.norm(self.W, dim=1, p=2)
            group_sparsity = (torch.sum(row_norms < 1e-8).item()) / self.W.shape[0]
            
            # Norms
            b_norm = torch.norm(self.B, p='fro').item()
            w_norm = torch.norm(self.W, p='fro').item()
            
            history["reconstruction_error"].append(error)
            history["relative_error"].append(relative_error)
            history["num_active_attributes"].append(self.k)
            history["group_sparsity"].append(group_sparsity)
            history["frobenius_norm_B"].append(b_norm)
            history["frobenius_norm_W"].append(w_norm)
            
            if verbose and t % 10 == 0:
                pbar.set_postfix({
                    "rel_error": f"{relative_error:.4f}",
                    "k": self.k,
                    "sparsity": f"{group_sparsity:.3f}"
                })
            
            # Convergence check
            if t > 50 and len(history["reconstruction_error"]) > 10:
                # Relative improvement over last 10 iterations
                recent_errors = history["reconstruction_error"][-10:]
                if len(recent_errors) >= 2:
                    rel_improvement = abs(recent_errors[-1] - recent_errors[0]) / (recent_errors[0] + 1e-12)
                    if rel_improvement < self.tol:
                        if verbose:
                            print(f"\nConverged at iteration {t} (rel_improvement={rel_improvement:.2e})")
                        break
            
            # Check for numerical issues
            if not torch.isfinite(self.B).all() or not torch.isfinite(self.W).all():
                if verbose:
                    print(f"\nNumerical instability detected at iteration {t}")
                break
        
        # Final statistics
        final_reconstruction = self.B @ self.W
        final_error = torch.norm(self.Y_normalized - final_reconstruction, p='fro').item()
        final_relative_error = final_error / np.sqrt(Y_norm_sq) if Y_norm_sq > 0 else final_error
        
        results = {
            "B": self.B.cpu(),
            "W": self.W.cpu(),
            "Y_normalized": self.Y_normalized.cpu(),
            "row_stats": self.row_stats,
            "history": history,
            "final_k": self.k,
            "final_error": final_error,
            "final_relative_error": final_relative_error,
            "final_group_sparsity": history["group_sparsity"][-1],
            "converged": t < self.max_iter - 1
        }
        
        return results
    
    def get_persona_importance(self) -> torch.Tensor:
        """
        Get importance scores for each persona.
        
        Returns:
            Persona importance scores (higher = more important)
        """
        if self.W is None:
            raise ValueError("Model not fitted yet")
        
        # L2 norm across attributes for each persona
        persona_norms = torch.norm(self.W, dim=0, p=2)
        return persona_norms
    
    def get_attribute_importance(self) -> torch.Tensor:
        """
        Get importance scores for each attribute.
        
        Returns:
            Attribute importance scores (higher = more important)
        """
        if self.W is None:
            raise ValueError("Model not fitted yet")
        
        # L2 norm across personas for each attribute
        attribute_norms = torch.norm(self.W, dim=1, p=2)
        return attribute_norms
    
    def get_selected_personas(self, top_k: Optional[int] = None, threshold: Optional[float] = None) -> torch.Tensor:
        """
        Get indices of selected personas based on importance.
        
        Args:
            top_k: Select top k personas by importance
            threshold: Select personas above threshold importance
            
        Returns:
            Indices of selected personas
        """
        if self.W is None:
            raise ValueError("Model not fitted yet")
        
        importance = self.get_persona_importance()
        
        if top_k is not None:
            _, selected = torch.topk(importance, min(top_k, len(importance)))
            return selected
        elif threshold is not None:
            selected = torch.where(importance > threshold)[0]
            return selected
        else:
            # Default: select personas with above-median importance
            median_importance = torch.median(importance)
            selected = torch.where(importance > median_importance)[0]
            return selected
    
    def reconstruct(self) -> torch.Tensor:
        """Reconstruct the reward matrix using learned B and W."""
        if self.B is None or self.W is None:
            raise ValueError("Model not fitted yet")
        
        return self.B @ self.W
    
    def interpret_attributes(self, top_questions: int = 5, question_names: Optional[list] = None) -> Dict:
        """
        Interpret learned attributes by finding top-loading questions.
        
        Args:
            top_questions: Number of top questions to show per attribute
            question_names: Optional list of question names/descriptions
            
        Returns:
            Dictionary with attribute interpretations
        """
        if self.B is None:
            raise ValueError("Model not fitted yet")
        
        interpretations = {}
        attribute_importance = self.get_attribute_importance()
        
        for attr_idx in range(self.k):
            # Get top questions for this attribute
            attr_loadings = torch.abs(self.B[:, attr_idx])
            top_indices = torch.topk(attr_loadings, min(top_questions, len(attr_loadings)))[1]
            
            attr_info = {
                "importance": attribute_importance[attr_idx].item(),
                "top_question_indices": top_indices.tolist(),
                "top_question_loadings": attr_loadings[top_indices].tolist()
            }
            
            # Add question names if provided
            if question_names:
                attr_info["top_questions"] = [
                    question_names[idx] if idx < len(question_names) else f"Question {idx}"
                    for idx in top_indices.tolist()
                ]
            
            interpretations[f"attribute_{attr_idx}"] = attr_info
        
        return interpretations
    
    def analyze_persona_attributes(self, persona_idx: int, top_attributes: int = 3) -> Dict:
        """
        Analyze which attributes are most important for a specific persona.
        
        Args:
            persona_idx: Index of persona to analyze
            top_attributes: Number of top attributes to show
            
        Returns:
            Dictionary with persona's top attributes
        """
        if self.W is None:
            raise ValueError("Model not fitted yet")
        
        if persona_idx >= self.W.shape[1]:
            raise ValueError(f"Persona index {persona_idx} out of range")
        
        # Get attribute weights for this persona
        persona_weights = torch.abs(self.W[:, persona_idx])
        top_indices = torch.topk(persona_weights, min(top_attributes, len(persona_weights)))[1]
        
        analysis = {
            "persona_idx": persona_idx,
            "total_importance": self.get_persona_importance()[persona_idx].item(),
            "top_attributes": [
                {
                    "attribute_idx": idx.item(),
                    "weight": persona_weights[idx].item(),
                    "relative_weight": (persona_weights[idx] / torch.sum(persona_weights)).item()
                }
                for idx in top_indices
            ]
        }
        
        return analysis
    
    def compute_persona_similarity(self, persona_idx1: int, persona_idx2: int) -> float:
        """
        Compute cosine similarity between two personas in attribute space.
        
        Args:
            persona_idx1: First persona index
            persona_idx2: Second persona index
            
        Returns:
            Cosine similarity (higher = more similar)
        """
        if self.W is None:
            raise ValueError("Model not fitted yet")
        
        w1 = self.W[:, persona_idx1]
        w2 = self.W[:, persona_idx2]
        
        # Cosine similarity
        dot_product = torch.dot(w1, w2)
        norm1 = torch.norm(w1, p=2)
        norm2 = torch.norm(w2, p=2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity.item()
    
    def get_reconstruction_quality(self) -> Dict:
        """
        Analyze reconstruction quality per question and persona.
        
        Returns:
            Dictionary with reconstruction analysis
        """
        if self.B is None or self.W is None:
            raise ValueError("Model not fitted yet")
        
        reconstruction = self.B @ self.W
        residual = self.Y_normalized - reconstruction
        
        # Per-question reconstruction error
        question_errors = torch.norm(residual, dim=1, p=2)
        
        # Per-persona reconstruction error  
        persona_errors = torch.norm(residual, dim=0, p=2)
        
        # Overall metrics
        total_error = torch.norm(residual, p='fro').item()
        
        analysis = {
            "total_frobenius_error": total_error,
            "mean_question_error": torch.mean(question_errors).item(),
            "std_question_error": torch.std(question_errors).item(),
            "mean_persona_error": torch.mean(persona_errors).item(),
            "std_persona_error": torch.std(persona_errors).item(),
            "worst_question_idx": torch.argmax(question_errors).item(),
            "worst_persona_idx": torch.argmax(persona_errors).item(),
            "best_question_idx": torch.argmin(question_errors).item(),
            "best_persona_idx": torch.argmin(persona_errors).item()
        }
        
        return analysis

    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            "B": self.B.cpu() if self.B is not None else None,
            "W": self.W.cpu() if self.W is not None else None,
            "Y_normalized": self.Y_normalized.cpu() if self.Y_normalized is not None else None,
            "row_stats": self.row_stats,
            "k": self.k,
            "k_init": self.k_init,
            "lmbda": self.lmbda,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "init_method": self.init_method,
            "normalize_rows": self.normalize_rows,
            "normalize_cols": self.normalize_cols
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.B = checkpoint["B"].to(self.device) if checkpoint["B"] is not None else None
        self.W = checkpoint["W"].to(self.device) if checkpoint["W"] is not None else None
        self.Y_normalized = checkpoint.get("Y_normalized")
        if self.Y_normalized is not None:
            self.Y_normalized = self.Y_normalized.to(self.device)
        self.row_stats = checkpoint.get("row_stats")
        self.k = checkpoint["k"]
        self.k_init = checkpoint.get("k_init", self.k)
        self.lmbda = checkpoint.get("lmbda", checkpoint.get("lambda21", 0.1))  # Backward compatibility
        self.beta = checkpoint["beta"]
        self.epsilon = checkpoint["epsilon"]
        self.init_method = checkpoint.get("init_method", "svd")
        self.normalize_rows = checkpoint.get("normalize_rows", True)
        self.normalize_cols = checkpoint.get("normalize_cols", True)