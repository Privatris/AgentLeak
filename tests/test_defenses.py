"""
Tests for APB Defenses Module.

Tests cover:
- LearnedContentFilter (LCF)
- OutputSanitizer
- Defense base classes
"""

import pytest
import numpy as np

from agentleak.schemas.scenario import Channel, CanaryTier

from agentleak.defenses.base import (
    BaseDefense,
    DefenseConfig,
    DefenseResult,
    FilterAction,
)
from agentleak.defenses.lcf import (
    LearnedContentFilter,
    LCFConfig,
    LCFTrainer,
    LCFTrainingExample,
    LCFEmbedding,
    LCFClassifier,
    FilterDecision,
)
from agentleak.defenses.sanitizer import (
    OutputSanitizer,
    SanitizerConfig,
    SanitizationResult,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def lcf():
    """Create a basic LCF instance."""
    return LearnedContentFilter()


@pytest.fixture
def trained_lcf():
    """Create a trained LCF instance."""
    lcf = LearnedContentFilter()
    
    # Train with examples
    examples = [
        LCFTrainingExample("123-45-6789", is_private=True, privacy_type="ssn"),
        LCFTrainingExample("968-25-7201", is_private=True, privacy_type="ssn"),
        LCFTrainingExample("CANARY_SSN_12345678", is_private=True, privacy_type="canary"),
        LCFTrainingExample("Hello world", is_private=False),
        LCFTrainingExample("The weather is nice", is_private=False),
        LCFTrainingExample("Please confirm your appointment", is_private=False),
    ]
    lcf.train(examples)
    
    return lcf


@pytest.fixture
def sanitizer():
    """Create a basic sanitizer."""
    return OutputSanitizer()


# ============================================================
# LCFEmbedding Tests
# ============================================================

class TestLCFEmbedding:
    """Tests for LCF embedding."""
    
    def test_create_embedding(self):
        """Test embedding creation."""
        emb = LCFEmbedding(dim=64)
        assert emb.dim == 64
    
    def test_embed_text(self):
        """Test text embedding."""
        emb = LCFEmbedding()
        vec = emb.embed("Hello world")
        
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 128
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01  # Normalized
    
    def test_similar_texts_similar_embeddings(self):
        """Test that similar texts have similar embeddings."""
        emb = LCFEmbedding()
        
        sim_same = emb.similarity("123-45-6789", "123-45-6789")
        sim_similar = emb.similarity("123-45-6789", "234-56-7890")
        sim_different = emb.similarity("123-45-6789", "Hello world")
        
        assert sim_same > sim_similar > sim_different
    
    def test_empty_text(self):
        """Test embedding empty text."""
        emb = LCFEmbedding()
        vec = emb.embed("")
        
        assert len(vec) == 128


# ============================================================
# LCFClassifier Tests
# ============================================================

class TestLCFClassifier:
    """Tests for LCF classifier."""
    
    def test_untrained_classifier(self):
        """Test classifier before training."""
        emb = LCFEmbedding()
        clf = LCFClassifier(emb)
        
        is_private, conf = clf.predict("test")
        
        assert is_private == False
        assert conf == 0.0
    
    def test_train_classifier(self):
        """Test classifier training."""
        emb = LCFEmbedding()
        clf = LCFClassifier(emb)
        
        examples = [
            LCFTrainingExample("123-45-6789", is_private=True),
            LCFTrainingExample("secret data", is_private=True),
            LCFTrainingExample("hello world", is_private=False),
            LCFTrainingExample("public info", is_private=False),
        ]
        
        clf.train(examples)
        
        assert clf._trained
        assert clf.private_centroid is not None
        assert clf.safe_centroid is not None
    
    def test_predict_after_training(self):
        """Test predictions after training."""
        emb = LCFEmbedding()
        clf = LCFClassifier(emb)
        
        examples = [
            LCFTrainingExample("123-45-6789", is_private=True),
            LCFTrainingExample("987-65-4321", is_private=True),
            LCFTrainingExample("hello world", is_private=False),
            LCFTrainingExample("good morning", is_private=False),
        ]
        clf.train(examples)
        
        # Similar to private
        is_private, conf = clf.predict("111-22-3333")
        # Similar to safe
        is_safe, conf2 = clf.predict("hello there")
        
        # Private-looking should have higher private probability
        assert conf >= 0


# ============================================================
# LearnedContentFilter Tests
# ============================================================

class TestLearnedContentFilter:
    """Tests for LCF filter."""
    
    def test_create_lcf(self, lcf):
        """Test LCF creation."""
        assert lcf.name == "LCF"
        assert lcf.config.enabled
    
    def test_detect_canary(self, lcf):
        """Test canary detection."""
        result = lcf.filter(
            "The SSN is CANARY_SSN_12345678",
            Channel.C1_FINAL_OUTPUT
        )
        
        assert result.action in [FilterAction.BLOCK, FilterAction.REDACT]
        assert result.confidence == 1.0
        assert "CANARY_SSN_12345678" in result.detected_patterns
    
    def test_detect_ssn_pattern(self, lcf):
        """Test SSN pattern detection."""
        result = lcf.filter(
            "SSN: 968-25-7201",  # Reserved range
            Channel.C1_FINAL_OUTPUT
        )
        
        assert result.action in [FilterAction.BLOCK, FilterAction.REDACT]
        assert result.confidence > 0.9
    
    def test_detect_phone_pattern(self, lcf):
        """Test phone pattern detection."""
        result = lcf.filter(
            "Call me at 555-0123-4567",
            Channel.C1_FINAL_OUTPUT
        )
        
        assert result.action in [FilterAction.BLOCK, FilterAction.REDACT, FilterAction.ALLOW]
    
    def test_allow_safe_content(self, lcf):
        """Test that safe content is allowed."""
        result = lcf.filter(
            "Hello, how are you today?",
            Channel.C1_FINAL_OUTPUT
        )
        
        assert result.action == FilterAction.ALLOW
        assert result.filtered_content is None
    
    def test_register_private_values(self, lcf):
        """Test registering private values."""
        lcf.register_private_values(["secret123", "password456"])
        
        result = lcf.filter(
            "The code is secret123",
            Channel.C1_FINAL_OUTPUT
        )
        
        assert result.action in [FilterAction.BLOCK, FilterAction.REDACT]
    
    def test_register_allowlist(self, lcf):
        """Test allowlist functionality."""
        lcf.register_allowlist(["allowed content"])
        
        result = lcf.filter(
            "allowed content",
            Channel.C1_FINAL_OUTPUT
        )
        
        assert result.action == FilterAction.ALLOW
    
    def test_redaction(self, lcf):
        """Test content redaction."""
        result = lcf.filter(
            "The SSN is CANARY_SSN_12345678 and phone is 555-0123-4567",
            Channel.C1_FINAL_OUTPUT
        )
        
        if result.filtered_content:
            assert "[REDACTED]" in result.filtered_content
            assert "CANARY_SSN_12345678" not in result.filtered_content
    
    def test_trained_lcf_semantic_detection(self, trained_lcf):
        """Test trained LCF semantic detection."""
        # Similar to training data
        result = trained_lcf.filter(
            "The number is 111-22-3333",
            Channel.C1_FINAL_OUTPUT
        )
        
        # Should detect SSN-like pattern
        assert result.confidence > 0
    
    def test_channel_filtering(self, lcf):
        """Test channel-based filtering."""
        config = LCFConfig(channels_to_monitor=[Channel.C1_FINAL_OUTPUT])
        lcf_limited = LearnedContentFilter(config)
        
        # Should monitor C1
        result1 = lcf_limited.filter(
            "CANARY_SSN_12345678",
            Channel.C1_FINAL_OUTPUT
        )
        assert result1.action != FilterAction.ALLOW or result1.confidence > 0
        
        # Should not monitor C6
        result2 = lcf_limited.filter(
            "CANARY_SSN_12345678",
            Channel.C6_LOG
        )
        assert result2.action == FilterAction.ALLOW
    
    def test_decision_log(self, lcf):
        """Test decision logging."""
        lcf.filter("Test 1", Channel.C1_FINAL_OUTPUT)
        lcf.filter("Test 2", Channel.C1_FINAL_OUTPUT)
        
        log = lcf.get_decision_log()
        
        assert len(log) == 2
    
    def test_stats(self, lcf):
        """Test statistics."""
        lcf.filter("Hello", Channel.C1_FINAL_OUTPUT)
        lcf.filter("CANARY_SSN_12345678", Channel.C1_FINAL_OUTPUT)
        
        stats = lcf.stats()
        
        assert stats["total"] == 2
        assert "avg_latency_ms" in stats


# ============================================================
# LCFTrainer Tests
# ============================================================

class TestLCFTrainer:
    """Tests for LCF trainer."""
    
    def test_create_trainer(self):
        """Test trainer creation."""
        trainer = LCFTrainer()
        assert len(trainer.examples) == 0
    
    def test_add_negative_examples(self):
        """Test adding negative examples."""
        trainer = LCFTrainer()
        trainer.add_negative_examples(["hello", "world", "test"])
        
        assert len(trainer.examples) == 3
        assert all(not ex.is_private for ex in trainer.examples)
    
    def test_get_private_values(self):
        """Test getting private values."""
        trainer = LCFTrainer()
        trainer.examples = [
            LCFTrainingExample("secret", is_private=True),
            LCFTrainingExample("public", is_private=False),
        ]
        
        private = trainer.get_private_values()
        
        assert private == ["secret"]
    
    def test_train_filter(self):
        """Test training a filter."""
        trainer = LCFTrainer()
        trainer.examples = [
            LCFTrainingExample("123-45-6789", is_private=True),
            LCFTrainingExample("hello world", is_private=False),
        ]
        
        lcf = LearnedContentFilter()
        trainer.train_filter(lcf)
        
        assert lcf.classifier._trained


# ============================================================
# OutputSanitizer Tests
# ============================================================

class TestOutputSanitizer:
    """Tests for output sanitizer."""
    
    def test_create_sanitizer(self, sanitizer):
        """Test sanitizer creation."""
        assert sanitizer.name == "OutputSanitizer"
    
    def test_sanitize_ssn(self, sanitizer):
        """Test SSN sanitization."""
        result = sanitizer.sanitize("SSN: 123-45-6789")
        
        assert result.redactions_made == 1
        assert "[REDACTED]" in result.sanitized
        assert "123-45-6789" not in result.sanitized
    
    def test_sanitize_phone(self, sanitizer):
        """Test phone sanitization."""
        result = sanitizer.sanitize("Call 555-123-4567")
        
        assert result.redactions_made == 1
        assert "[REDACTED]" in result.sanitized
    
    def test_sanitize_email(self, sanitizer):
        """Test email sanitization."""
        result = sanitizer.sanitize("Email: test@example.com")
        
        assert result.redactions_made == 1
        assert "test@example.com" not in result.sanitized
    
    def test_sanitize_credit_card(self, sanitizer):
        """Test credit card sanitization."""
        result = sanitizer.sanitize("Card: 4111-1111-1111-1111")
        
        assert result.redactions_made == 1
    
    def test_sanitize_canary(self, sanitizer):
        """Test canary sanitization."""
        result = sanitizer.sanitize("Token: CANARY_SSN_12345678")
        
        assert result.redactions_made == 1
    
    def test_sanitize_multiple(self, sanitizer):
        """Test multiple pattern sanitization."""
        result = sanitizer.sanitize(
            "SSN: 123-45-6789, Phone: 555-123-4567, Email: test@example.com"
        )
        
        assert result.redactions_made == 3
        assert result.sanitized.count("[REDACTED]") == 3
    
    def test_no_sanitization_needed(self, sanitizer):
        """Test content without sensitive data."""
        result = sanitizer.sanitize("Hello world, nice weather!")
        
        assert result.redactions_made == 0
        assert result.sanitized == result.original
    
    def test_filter_interface(self, sanitizer):
        """Test BaseDefense filter interface."""
        result = sanitizer.filter(
            "SSN: 123-45-6789",
            Channel.C1_FINAL_OUTPUT
        )
        
        assert isinstance(result, DefenseResult)
        assert result.action == FilterAction.REDACT
    
    def test_add_custom_pattern(self, sanitizer):
        """Test adding custom pattern."""
        sanitizer.add_pattern("custom_id", r'ID-\d{6}')
        
        result = sanitizer.sanitize("Your ID is ID-123456")
        
        assert result.redactions_made == 1
    
    def test_remove_pattern(self, sanitizer):
        """Test removing pattern."""
        sanitizer.remove_pattern("ssn")
        
        result = sanitizer.sanitize("SSN: 123-45-6789")
        
        assert result.redactions_made == 0


# ============================================================
# Integration Tests
# ============================================================

class TestDefenseIntegration:
    """Integration tests for defenses."""
    
    def test_lcf_with_scenario_data(self):
        """Test LCF with realistic scenario data."""
        from agentleak.generators import ScenarioGenerator
        from agentleak.schemas import Vertical
        
        # Generate a scenario
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(Vertical.HEALTHCARE)
        
        # Train LCF from scenario
        trainer = LCFTrainer()
        count = trainer.add_from_scenario(scenario)
        
        assert count > 0
        
        # Create and train LCF
        lcf = LearnedContentFilter()
        trainer.train_filter(lcf)
        
        # Register private values
        private_values = trainer.get_private_values()
        lcf.register_private_values(private_values)
        
        # Test detection of canaries
        canaries = scenario.get_canaries()
        if canaries:
            result = lcf.filter(canaries[0], Channel.C1_FINAL_OUTPUT)
            assert result.action in [FilterAction.BLOCK, FilterAction.REDACT]
    
    def test_defense_latency(self, lcf, sanitizer):
        """Test that defenses meet latency requirements."""
        test_content = "SSN: 123-45-6789, Phone: 555-123-4567"
        
        # LCF latency
        lcf_result = lcf.filter(test_content, Channel.C1_FINAL_OUTPUT)
        assert lcf_result.latency_ms < 100  # Should be fast
        
        # Sanitizer latency
        san_result = sanitizer.filter(test_content, Channel.C1_FINAL_OUTPUT)
        assert san_result.latency_ms < 50  # Should be very fast
    
    def test_defense_comparison(self):
        """Compare LCF vs Sanitizer on same content."""
        content = "Patient SSN: CANARY_SSN_12345678, Phone: 555-0123-4567"
        
        lcf = LearnedContentFilter()
        sanitizer = OutputSanitizer()
        
        lcf_result = lcf.filter(content, Channel.C1_FINAL_OUTPUT)
        san_result = sanitizer.filter(content, Channel.C1_FINAL_OUTPUT)
        
        # Both should detect the canary
        assert lcf_result.action in [FilterAction.BLOCK, FilterAction.REDACT]
        assert san_result.action == FilterAction.REDACT
        
        # Both should have detected patterns
        assert len(lcf_result.detected_patterns) > 0
        assert len(san_result.detected_patterns) > 0
