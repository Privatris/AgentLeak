"""
Tests for agentleak Defenses Module.

Tests cover:
- OutputSanitizer
- Defense base classes
"""

import pytest

from agentleak.schemas.scenario import Channel

from agentleak.defenses.base import (
    BaseDefense,
    DefenseConfig,
    DefenseResult,
    FilterAction,
)
from agentleak.defenses.sanitizer import (
    OutputSanitizer,
    SanitizerConfig,
    SanitizationResult,
)


@pytest.fixture
def sanitizer():
    """Create a basic sanitizer."""
    return OutputSanitizer()


# ============================================================
# OutputSanitizer Tests
# ============================================================

class TestOutputSanitizer:
    """Tests for output sanitizer."""
    
    def test_create_sanitizer(self, sanitizer):
        """Test sanitizer creation."""
        assert sanitizer is not None
        assert hasattr(sanitizer, 'sanitize')
    
    def test_sanitize_ssn(self, sanitizer):
        """Test SSN sanitization."""
        text = "The SSN is 123-45-6789"
        result = sanitizer.sanitize(text)
        
        assert isinstance(result, SanitizationResult)
        if result.sanitized:
            assert "123-45-6789" not in result.sanitized or "[REDACTED]" in result.sanitized
    
    def test_sanitize_credit_card(self, sanitizer):
        """Test credit card sanitization."""
        text = "Card: 4111-1111-1111-1111"
        result = sanitizer.sanitize(text)
        
        assert isinstance(result, SanitizationResult)
        if result.sanitized:
            assert "4111-1111-1111-1111" not in result.sanitized or "[REDACTED]" in result.sanitized
    
    def test_safe_content_passthrough(self, sanitizer):
        """Test safe content is not modified."""
        text = "Hello, how are you?"
        result = sanitizer.sanitize(text)
        
        assert isinstance(result, SanitizationResult)
        # Safe content should have 0 redactions
        assert result.redactions_made == 0


class TestBaseDefense:
    """Tests for base defense class."""
    
    def test_defense_config(self):
        """Test defense configuration."""
        config = DefenseConfig(enabled=True)
        assert config.enabled
    
    def test_filter_action_enum(self):
        """Test filter action enum."""
        assert FilterAction.ALLOW
        assert FilterAction.BLOCK
        assert FilterAction.REDACT
