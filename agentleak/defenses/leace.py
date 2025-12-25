"""
LEACE (Linear Erasure of Attribute by Concept Erasure) Implementation.

This module implements the LEACE projection as described in:
    Belrose et al., 2023, "LEACE: Perfect linear concept erasure in closed form"

LEACE provides:
- P = I - UU^T projection to remove linearly-encoded sensitive attributes
- Orthonormal basis U of the sensitive subspace
- Variance-based leakage quantification

Key equations:
- Sensitive subspace: span of class-discriminating directions
- Projection: P = I - U @ U.T (orthogonal complement)
- Leakage: ||U^T z||^2 (variance in sensitive subspace)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class LEACEProjection:
    """
    LEACE projection result.
    
    Attributes:
        P: (d, d) projection matrix
        U: (d, k) orthonormal basis of sensitive subspace
        explained_variance: Variance explained by sensitive directions
        rank: Effective rank of sensitive subspace
    """
    P: np.ndarray
    U: np.ndarray
    explained_variance: float
    rank: int
    
    def project(self, z: np.ndarray) -> np.ndarray:
        """
        Apply LEACE projection to embedding.
        
        Args:
            z: (d,) or (n, d) embedding vector(s)
        
        Returns:
            Projected embedding(s) with sensitive information removed
        """
        return z @ self.P.T if z.ndim == 2 else self.P @ z
    
    def leakage(self, z: np.ndarray) -> float:
        """
        Compute leakage (sensitive component norm).
        
        Args:
            z: (d,) embedding vector
        
        Returns:
            L2 norm of projection onto sensitive subspace: ||U^T z||
        """
        sensitive_component = self.U.T @ z
        return float(np.linalg.norm(sensitive_component))
    
    def variance_cost(self, z: np.ndarray) -> float:
        """
        Compute variance cost for Cumulative Variance Budget.
        
        Args:
            z: (d,) embedding vector
        
        Returns:
            Squared L2 norm: ||U^T z||^2
        """
        leak = self.leakage(z)
        return leak * leak


def compute_leace_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    regularization: float = 1e-6,
) -> LEACEProjection:
    """
    Compute LEACE projection matrix from labeled data.
    
    Implements Linear Concept Erasure:
    1. Compute class means and within-class covariance
    2. Find directions that separate classes (between-class scatter)
    3. Return P = I - UU^T where U spans the sensitive subspace
    
    Args:
        embeddings: (n, d) matrix of embedding vectors
        labels: (n,) array of binary labels (0 = safe, 1 = private)
        regularization: Regularization for numerical stability
        
    Returns:
        LEACEProjection containing P, U, and metadata
    
    References:
        Belrose et al., 2023. "LEACE: Perfect linear concept erasure in closed form"
    """
    n, d = embeddings.shape
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    if k < 2:
        # Cannot compute projection with only one class
        return LEACEProjection(
            P=np.eye(d),
            U=np.zeros((d, 0)),
            explained_variance=0.0,
            rank=0
        )
    
    # Compute class means
    class_means = np.zeros((k, d))
    class_counts = np.zeros(k)
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_means[i] = embeddings[mask].mean(axis=0)
        class_counts[i] = mask.sum()
    
    # Global mean
    global_mean = embeddings.mean(axis=0)
    
    # Compute within-class covariance (for regularization)
    Sigma_W = np.zeros((d, d))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centered = embeddings[mask] - class_means[i]
        Sigma_W += centered.T @ centered
    
    Sigma_W /= max(1, n - k)
    Sigma_W += regularization * np.eye(d)
    
    # Weighted centered class means for between-class scatter
    sqrt_weights = np.sqrt(class_counts / n)
    centered_means = (class_means - global_mean) * sqrt_weights[:, np.newaxis]
    
    # SVD to get principal directions of between-class scatter
    _, S, Vt = np.linalg.svd(centered_means, full_matrices=False)
    
    # Total variance for explained ratio
    total_variance = np.sum(S ** 2) if len(S) > 0 else 0.0
    
    # Keep significant directions (eigenvalues > regularization)
    significant = S > regularization
    U = Vt[significant].T  # (d, k_eff)
    
    # Explained variance by kept directions
    explained = np.sum(S[significant] ** 2) if np.any(significant) else 0.0
    explained_ratio = explained / max(total_variance, regularization)
    
    # Orthonormalize (should already be from SVD, but ensure)
    if U.shape[1] > 0:
        U, _ = np.linalg.qr(U)
    else:
        U = np.zeros((d, 0))
    
    # Projection matrix: P = I - UU^T
    P = np.eye(d) - U @ U.T
    
    return LEACEProjection(
        P=P,
        U=U,
        explained_variance=float(explained_ratio),
        rank=U.shape[1]
    )


def train_leace_from_examples(
    embedding_fn,
    private_texts: list[str],
    safe_texts: list[str],
) -> LEACEProjection:
    """
    Train LEACE projection from text examples.
    
    Args:
        embedding_fn: Function to embed text (str -> np.ndarray)
        private_texts: List of private text examples
        safe_texts: List of safe text examples
    
    Returns:
        LEACEProjection ready for use
    """
    # Embed all examples
    all_texts = private_texts + safe_texts
    embeddings = np.array([embedding_fn(t) for t in all_texts])
    
    # Labels: 1 = private, 0 = safe
    labels = np.array([1] * len(private_texts) + [0] * len(safe_texts))
    
    return compute_leace_projection(embeddings, labels)


class LEACEFilter:
    """
    LEACE-based content filter.
    
    Uses LEACE projection to detect and optionally remove
    sensitive information from embeddings.
    """
    
    def __init__(
        self,
        embedding_fn,
        leakage_threshold: float = 0.3,
    ):
        """
        Initialize LEACE filter.
        
        Args:
            embedding_fn: Function to embed text
            leakage_threshold: Leakage threshold for flagging content
        """
        self.embedding_fn = embedding_fn
        self.leakage_threshold = leakage_threshold
        self.projection: Optional[LEACEProjection] = None
        self._trained = False
    
    def train(
        self,
        private_texts: list[str],
        safe_texts: list[str],
    ) -> None:
        """
        Train the filter from examples.
        
        Args:
            private_texts: Examples of private content
            safe_texts: Examples of safe content
        """
        self.projection = train_leace_from_examples(
            self.embedding_fn,
            private_texts,
            safe_texts,
        )
        self._trained = True
    
    def is_private(self, text: str) -> tuple[bool, float]:
        """
        Check if text is likely private.
        
        Args:
            text: Text to check
        
        Returns:
            (is_private, leakage_score)
        """
        if not self._trained or self.projection is None:
            return False, 0.0
        
        z = self.embedding_fn(text)
        leakage = self.projection.leakage(z)
        
        return leakage > self.leakage_threshold, leakage
    
    def filter(self, text: str, embedding: np.ndarray) -> np.ndarray:
        """
        Apply LEACE projection to remove sensitive components.
        
        Args:
            text: Original text (for reference)
            embedding: Embedding to filter
        
        Returns:
            Filtered embedding
        """
        if not self._trained or self.projection is None:
            return embedding
        
        return self.projection.project(embedding)
