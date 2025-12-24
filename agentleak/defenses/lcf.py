"""
AgentLeak Learned Content Filter (LCF) - ML-based privacy filter.

LCF is a learned defense that classifies content as private or safe
based on training examples. It uses:
1. Pattern matching for known private data formats
2. LEACE projection for removing linear sensitive information
3. Embedding similarity to detect semantic leakage
4. Ensemble of simple classifiers for robustness

The filter is designed to have:
- High recall (catch most leaks)
- Acceptable precision (minimize false positives)
- Low latency (< 50ms per decision)

References:
    Belrose et al., 2023. "LEACE: Perfect linear concept erasure in closed form"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import re
import time
import numpy as np

from ..schemas.scenario import Channel, CanaryTier
from .base import BaseDefense, DefenseConfig, DefenseResult, FilterAction
from .leace import LEACEProjection, compute_leace_projection


class FilterDecision(str, Enum):
    """LCF filter decision types."""
    SAFE = "safe"
    CANARY_DETECTED = "canary_detected"
    PATTERN_MATCH = "pattern_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LEACE_PROJECTION = "leace_projection"
    ENSEMBLE_VOTE = "ensemble_vote"


@dataclass
class LCFConfig(DefenseConfig):
    """Configuration for Learned Content Filter."""
    
    # Pattern detection
    enable_pattern_matching: bool = True
    pattern_types: list[str] = field(default_factory=lambda: ["ssn", "phone", "cc", "email"])
    
    # LEACE projection (removes linearly-encoded sensitive attributes)
    enable_leace: bool = True
    leace_threshold: float = 0.30  # Leakage threshold for flagging
    
    # Semantic detection
    enable_semantic_filter: bool = True
    semantic_threshold: float = 0.80
    
    # Canary detection
    enable_canary_detection: bool = True
    
    # Ensemble settings
    enable_ensemble: bool = True
    ensemble_threshold: float = 0.5  # Majority vote threshold
    
    # Action settings
    default_action: FilterAction = FilterAction.REDACT
    high_confidence_threshold: float = 0.9
    
    # Performance
    max_latency_ms: float = 50.0


@dataclass
class LCFTrainingExample:
    """A training example for LCF."""
    text: str
    is_private: bool
    privacy_type: Optional[str] = None  # e.g., "ssn", "diagnosis"
    weight: float = 1.0


class LCFEmbedding:
    """
    Simple character-frequency based embedding for LCF.
    
    This is a lightweight alternative to transformer embeddings
    that works well for detecting structured private data.
    """
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._char_map: dict[str, int] = {}
        self._build_char_map()
    
    def _build_char_map(self) -> None:
        """Build character to index mapping."""
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 .-_@#$%"
        for i, c in enumerate(chars):
            self._char_map[c] = i
    
    def embed(self, text: str) -> np.ndarray:
        """Embed text as frequency vector."""
        text = text.lower()
        vec = np.zeros(self.dim)
        
        for char in text:
            if char in self._char_map:
                idx = self._char_map[char] % self.dim
                vec[idx] += 1
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        
        # Add n-gram features
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            idx = (ord(bigram[0]) * 31 + ord(bigram[1])) % self.dim
            vec[idx] += 0.5
        
        # Re-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        
        return vec
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between texts."""
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        return float(np.dot(v1, v2))


class LCFClassifier:
    """
    Simple trainable classifier for LCF.
    
    Uses a centroid-based approach:
    - Computes centroid of private examples
    - Computes centroid of safe examples
    - Classifies based on distance to each
    """
    
    def __init__(self, embedding: LCFEmbedding, threshold: float = 0.65):
        self.embedding = embedding
        self.private_centroid: Optional[np.ndarray] = None
        self.safe_centroid: Optional[np.ndarray] = None
        self.threshold: float = threshold  # Raised from 0.5 to reduce false positives
        self._trained = False
        self._private_examples: list[np.ndarray] = []  # Store for variance calculation
    
    def train(self, examples: list[LCFTrainingExample]) -> None:
        """Train classifier on examples."""
        private_vecs = []
        safe_vecs = []
        
        for ex in examples:
            vec = self.embedding.embed(ex.text)
            if ex.is_private:
                private_vecs.append(vec * ex.weight)
            else:
                safe_vecs.append(vec * ex.weight)
        
        # Store for later analysis
        self._private_examples = private_vecs.copy()
        
        if private_vecs:
            self.private_centroid = np.mean(private_vecs, axis=0)
            norm = np.linalg.norm(self.private_centroid)
            if norm > 0:
                self.private_centroid /= norm
        
        if safe_vecs:
            self.safe_centroid = np.mean(safe_vecs, axis=0)
            norm = np.linalg.norm(self.safe_centroid)
            if norm > 0:
                self.safe_centroid /= norm
        
        self._trained = True
    
    def predict(self, text: str) -> tuple[bool, float]:
        """
        Predict if text is private.
        
        Returns:
            (is_private, confidence)
        """
        if not self._trained:
            return False, 0.0
        
        # Cannot classify without both centroids
        if self.private_centroid is None or self.safe_centroid is None:
            return False, 0.0  # Default to safe when we can't distinguish
        
        # Short generic text is unlikely to be private
        if len(text.strip()) < 10:
            return False, 0.8
        
        vec = self.embedding.embed(text)
        
        # Distance to each centroid (cosine similarity)
        dist_private = float(np.dot(vec, self.private_centroid))
        dist_safe = float(np.dot(vec, self.safe_centroid))
        
        # Compute relative probability
        # If safe distance is much higher, bias toward safe classification
        if dist_private + dist_safe > 0:
            prob_private = dist_private / (dist_private + dist_safe + 1e-10)
        else:
            prob_private = 0.5
        
        # Apply margin: require clear separation to classify as private
        margin = dist_private - dist_safe
        is_private = prob_private > self.threshold and margin > 0.1
        confidence = abs(prob_private - 0.5) * 2  # Scale to [0, 1]
        
        return is_private, confidence


class LearnedContentFilter(BaseDefense):
    """
    Learned Content Filter - ML-based privacy defense.
    
    Combines multiple detection methods:
    1. Regex patterns for structured data (SSN, phone, etc.)
    2. Canary token detection (CANARY_* patterns)
    3. Embedding similarity for semantic matching
    4. Trained classifier ensemble
    
    Example:
        lcf = LearnedContentFilter()
        
        # Train on examples
        lcf.train([
            LCFTrainingExample("SSN: 123-45-6789", is_private=True),
            LCFTrainingExample("Hello world", is_private=False),
        ])
        
        # Or register known private values
        lcf.register_private_values(["123-45-6789", "secret diagnosis"])
        
        # Filter content
        result = lcf.filter("The SSN is 123-45-6789", Channel.C1_FINAL_OUTPUT)
        if result.blocked:
            print("Blocked private content!")
    """
    
    # Common PII patterns
    PATTERNS = {
        "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "phone": re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),
        "cc": re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "canary": re.compile(r'CANARY_[A-Z]+_[A-Z0-9]{8}'),
    }
    
    # Reserved ranges (high confidence private)
    SSN_RESERVED_RANGE = range(900, 1000)  # 900-999 prefix
    PHONE_RESERVED_PREFIX = "555-01"
    CC_RESERVED_PREFIX = "4111"
    
    def __init__(self, config: Optional[LCFConfig] = None):
        super().__init__(config or LCFConfig())
        self.config: LCFConfig = self.config
        
        # Initialize components
        self.embedding = LCFEmbedding()
        self.classifier = LCFClassifier(self.embedding)
        
        # LEACE projection (trained separately)
        self._leace_projection: Optional[LEACEProjection] = None
        
        # Known private values for exact matching
        self._private_values: set[str] = set()
        self._private_embeddings: list[tuple[str, np.ndarray]] = []
        
        # Allowlist for false positive reduction
        self._allowlist: set[str] = set()
    
    @property
    def name(self) -> str:
        return "LCF"
    
    def register_private_values(self, values: list[str]) -> None:
        """
        Register known private values for exact matching.
        
        Args:
            values: List of private values to detect
        """
        for value in values:
            self._private_values.add(value.lower())
            vec = self.embedding.embed(value)
            self._private_embeddings.append((value, vec))
    
    def register_allowlist(self, values: list[str]) -> None:
        """Register values that should never be flagged."""
        for value in values:
            self._allowlist.add(value.lower())
    
    def train(self, examples: list[LCFTrainingExample]) -> None:
        """
        Train the LCF classifier on examples.
        
        Args:
            examples: Training examples
        """
        self.classifier.train(examples)
        
        # Also register private values for exact matching
        for ex in examples:
            if ex.is_private:
                self._private_values.add(ex.text.lower())
                vec = self.embedding.embed(ex.text)
                self._private_embeddings.append((ex.text, vec))
        
        # Train LEACE projection if enabled
        if self.config.enable_leace and len(examples) >= 10:
            self._train_leace(examples)
    
    def _train_leace(self, examples: list[LCFTrainingExample]) -> None:
        """
        Train LEACE projection from examples.
        
        LEACE (Linear Erasure of Attribute by Concept Erasure) computes
        a projection matrix P = I - UU^T that removes linearly-encoded
        sensitive information from embeddings.
        
        Reference:
            Belrose et al., 2023. "LEACE: Perfect linear concept erasure in closed form"
        """
        embeddings = np.array([self.embedding.embed(ex.text) for ex in examples])
        labels = np.array([1 if ex.is_private else 0 for ex in examples])
        
        self._leace_projection = compute_leace_projection(embeddings, labels)
    
    def _check_leace(self, content: str) -> tuple[bool, float]:
        """
        Check content using LEACE projection.
        
        Returns:
            (is_private, leakage_score)
        """
        if self._leace_projection is None:
            return False, 0.0
        
        vec = self.embedding.embed(content)
        leakage = self._leace_projection.leakage(vec)
        
        return leakage > self.config.leace_threshold, leakage
    
    def filter(self, content: str, channel: Channel) -> DefenseResult:
        """
        Filter content for private information.
        
        Returns DefenseResult with action and details.
        """
        start_time = time.time()
        
        if not self.should_monitor(channel):
            return DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
                reason="Channel not monitored",
            )
        
        # Check allowlist first
        if content.lower() in self._allowlist:
            return DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
                reason="In allowlist",
            )
        
        detected_patterns: list[str] = []
        max_confidence = 0.0
        decision = FilterDecision.SAFE
        
        # 1. Canary detection (highest priority)
        if self.config.enable_canary_detection:
            canary_matches = self.PATTERNS["canary"].findall(content)
            if canary_matches:
                detected_patterns.extend(canary_matches)
                max_confidence = 1.0
                decision = FilterDecision.CANARY_DETECTED
        
        # 2. Pattern matching
        if self.config.enable_pattern_matching and decision == FilterDecision.SAFE:
            for pattern_type in self.config.pattern_types:
                if pattern_type in self.PATTERNS:
                    matches = self.PATTERNS[pattern_type].findall(content)
                    if matches:
                        # Check if reserved range
                        is_reserved = self._check_reserved(matches, pattern_type)
                        if is_reserved:
                            detected_patterns.extend(matches)
                            max_confidence = max(max_confidence, 0.95)
                            decision = FilterDecision.PATTERN_MATCH
                        else:
                            # Lower confidence for non-reserved
                            detected_patterns.extend(matches)
                            max_confidence = max(max_confidence, 0.7)
                            decision = FilterDecision.PATTERN_MATCH
        
        # 3. Exact value matching
        if decision == FilterDecision.SAFE:
            content_lower = content.lower()
            for private_val in self._private_values:
                if private_val in content_lower:
                    detected_patterns.append(private_val)
                    max_confidence = max(max_confidence, 0.9)
                    decision = FilterDecision.SEMANTIC_SIMILARITY
        
        # 4. Semantic similarity
        if self.config.enable_semantic_filter and decision == FilterDecision.SAFE:
            for orig_text, orig_vec in self._private_embeddings:
                sim = self.embedding.similarity(content, orig_text)
                if sim > self.config.semantic_threshold:
                    detected_patterns.append(f"similar_to:{orig_text[:20]}...")
                    max_confidence = max(max_confidence, sim)
                    decision = FilterDecision.SEMANTIC_SIMILARITY
                    break
        
        # 5. LEACE projection (removes linearly-encoded sensitive attributes)
        if self.config.enable_leace and decision == FilterDecision.SAFE:
            is_leace_private, leakage = self._check_leace(content)
            if is_leace_private:
                detected_patterns.append(f"leace_leakage:{leakage:.3f}")
                max_confidence = max(max_confidence, min(0.95, 0.5 + leakage))
                decision = FilterDecision.LEACE_PROJECTION
        
        # 6. Classifier ensemble
        if self.config.enable_ensemble and self.classifier._trained and decision == FilterDecision.SAFE:
            is_private, conf = self.classifier.predict(content)
            if is_private and conf > self.config.ensemble_threshold:
                detected_patterns.append("classifier")
                max_confidence = max(max_confidence, conf)
                decision = FilterDecision.ENSEMBLE_VOTE
        
        # Determine action
        if decision == FilterDecision.SAFE:
            action = FilterAction.ALLOW
            filtered_content = None
            reason = None
        elif max_confidence >= self.config.high_confidence_threshold:
            action = FilterAction.BLOCK if self.config.strict_mode else FilterAction.REDACT
            filtered_content = self._redact(content, detected_patterns)
            reason = f"High confidence {decision.value}: {detected_patterns}"
        else:
            action = self.config.default_action
            filtered_content = self._redact(content, detected_patterns) if action == FilterAction.REDACT else None
            reason = f"{decision.value}: {detected_patterns}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = DefenseResult(
            action=action,
            original_content=content,
            filtered_content=filtered_content,
            confidence=max_confidence,
            reason=reason,
            detected_patterns=detected_patterns,
            latency_ms=latency_ms,
        )
        
        self._log_decision(result)
        return result
    
    def _check_reserved(self, matches: list[str], pattern_type: str) -> bool:
        """Check if matches are in reserved/test ranges."""
        for match in matches:
            if pattern_type == "ssn":
                prefix = int(match.split("-")[0])
                if prefix in self.SSN_RESERVED_RANGE:
                    return True
            elif pattern_type == "phone":
                if self.PHONE_RESERVED_PREFIX in match:
                    return True
            elif pattern_type == "cc":
                if match.replace("-", "").replace(" ", "").startswith(self.CC_RESERVED_PREFIX):
                    return True
        return False
    
    def _redact(self, content: str, patterns: list[str]) -> str:
        """Redact detected patterns from content."""
        result = content
        for pattern in patterns:
            if pattern.startswith("similar_to:") or pattern == "classifier":
                continue
            result = result.replace(pattern, "[REDACTED]")
        
        # Also redact regex matches
        for pattern_type, regex in self.PATTERNS.items():
            result = regex.sub("[REDACTED]", result)
        
        return result


class LCFTrainer:
    """
    Helper class for training LCF from scenarios.
    
    Extracts training examples from AgentLeak scenarios.
    """
    
    def __init__(self):
        self.examples: list[LCFTrainingExample] = []
    
    def add_from_scenario(self, scenario: Any) -> int:
        """
        Extract training examples from a scenario.
        
        Returns number of examples added.
        """
        count = 0
        
        # Private examples from vault
        if hasattr(scenario, 'private_vault'):
            for record in scenario.private_vault.records:
                for field_name, value in record.fields.items():
                    if isinstance(value, str):
                        tier = record.canary_tier if hasattr(record, 'canary_tier') else None
                        weight = 1.0
                        if tier == CanaryTier.OBVIOUS:
                            weight = 2.0  # Higher weight for obvious canaries
                        
                        self.examples.append(LCFTrainingExample(
                            text=value,
                            is_private=True,
                            privacy_type=field_name,
                            weight=weight,
                        ))
                        count += 1
        
        # Safe examples from allowed set (just field names, not values)
        # AllowedSet only contains field paths, not actual values
        # So we skip this for training - allowed set is for validation, not data
        
        return count
    
    def add_negative_examples(self, texts: list[str]) -> None:
        """Add safe (negative) examples."""
        for text in texts:
            self.examples.append(LCFTrainingExample(
                text=text,
                is_private=False,
            ))
    
    def train_filter(self, lcf: LearnedContentFilter) -> None:
        """Train an LCF instance with collected examples."""
        if self.examples:
            lcf.train(self.examples)
    
    def get_private_values(self) -> list[str]:
        """Get list of all private values for registration."""
        return [ex.text for ex in self.examples if ex.is_private]
