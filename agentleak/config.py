"""
AgentLeak Configuration Module.

Loads configuration from environment variables and .env file.
Provides type-safe access to all configuration values.

Usage:
    from agentleak.config import config
    
    api_key = config.openai_api_key
    model = config.default_model
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load .env file if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        env_path = Path.cwd() / ".env"
    
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            logger.debug(f"Loaded environment from {env_path}")
        except Exception as e:
            logger.warning(f"Failed to load .env: {e}")


# Load .env on import
_load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default


def _get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class AgentLeakConfig:
    """Central configuration for AgentLeak benchmark."""
    
    # =========================================================================
    # LLM API Keys
    # =========================================================================
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API key."""
        key = _get_env("OPENAI_API_KEY")
        return key if key else None
    
    @property
    def openai_org_id(self) -> Optional[str]:
        """OpenAI organization ID."""
        org = _get_env("OPENAI_ORG_ID")
        return org if org else None
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Anthropic API key."""
        key = _get_env("ANTHROPIC_API_KEY")
        return key if key else None
    
    @property
    def azure_openai_api_key(self) -> Optional[str]:
        """Azure OpenAI API key."""
        key = _get_env("AZURE_OPENAI_API_KEY")
        return key if key else None
    
    @property
    def azure_openai_endpoint(self) -> Optional[str]:
        """Azure OpenAI endpoint."""
        endpoint = _get_env("AZURE_OPENAI_ENDPOINT")
        return endpoint if endpoint else None
    
    @property
    def azure_openai_deployment(self) -> str:
        """Azure OpenAI deployment name."""
        return _get_env("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    @property
    def google_api_key(self) -> Optional[str]:
        """Google AI API key."""
        key = _get_env("GOOGLE_API_KEY")
        return key if key else None
    
    @property
    def huggingface_api_key(self) -> Optional[str]:
        """Hugging Face API key."""
        key = _get_env("HUGGINGFACE_API_KEY")
        return key if key else None
    
    # =========================================================================
    # Model Configuration
    # =========================================================================
    @property
    def default_model(self) -> str:
        """Default LLM model to use."""
        return _get_env("DEFAULT_MODEL", "gpt-4o-mini")
    
    @property
    def temperature(self) -> float:
        """LLM temperature setting."""
        return _get_env_float("LLM_TEMPERATURE", 0.0)
    
    @property
    def max_tokens(self) -> int:
        """Maximum tokens per LLM response."""
        return _get_env_int("LLM_MAX_TOKENS", 4096)
    
    # =========================================================================
    # Benchmark Settings
    # =========================================================================
    @property
    def n_scenarios(self) -> int:
        """Number of scenarios to generate."""
        return _get_env_int("BENCHMARK_N_SCENARIOS", 1000)
    
    @property
    def attack_probability(self) -> float:
        """Probability of applying attack to scenario."""
        return _get_env_float("BENCHMARK_ATTACK_PROB", 0.5)
    
    @property
    def seed(self) -> int:
        """Random seed for reproducibility."""
        return _get_env_int("BENCHMARK_SEED", 42)
    
    # =========================================================================
    # Detection Settings
    # =========================================================================
    @property
    def embedding_model(self) -> str:
        """Sentence transformer model for embeddings."""
        return _get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    @property
    def semantic_threshold(self) -> float:
        """Threshold for semantic similarity detection."""
        return _get_env_float("SEMANTIC_THRESHOLD", 0.85)
    
    # =========================================================================
    # Defense Settings (LCF)
    # =========================================================================
    @property
    def lcf_rank(self) -> int:
        """Rank for LEACE-style filtering in LCF."""
        return _get_env_int("LCF_RANK", 64)
    
    @property
    def lcf_alpha(self) -> float:
        """Regularization alpha for LCF."""
        return _get_env_float("LCF_ALPHA", 0.01)
    
    # =========================================================================
    # Output Settings
    # =========================================================================
    @property
    def output_dir(self) -> Path:
        """Directory for benchmark outputs."""
        return Path(_get_env("OUTPUT_DIR", "./agentleak_data"))
    
    @property
    def verbose(self) -> bool:
        """Enable verbose logging."""
        return _get_env_bool("VERBOSE", True)
    
    @property
    def save_traces(self) -> bool:
        """Save execution traces."""
        return _get_env_bool("SAVE_TRACES", True)
    
    @property
    def debug(self) -> bool:
        """Enable debug mode."""
        return _get_env_bool("DEBUG", False)
    
    @property
    def log_level(self) -> str:
        """Logging level."""
        return _get_env("LOG_LEVEL", "INFO")
    
    # =========================================================================
    # Validation
    # =========================================================================
    def validate(self) -> list[str]:
        """Validate configuration, return list of warnings."""
        warnings = []
        
        if not self.openai_api_key and not self.anthropic_api_key:
            warnings.append(
                "No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )
        
        if self.semantic_threshold < 0.5 or self.semantic_threshold > 1.0:
            warnings.append(
                f"SEMANTIC_THRESHOLD={self.semantic_threshold} is unusual. "
                "Expected 0.5-1.0"
            )
        
        return warnings
    
    def has_llm_access(self) -> bool:
        """Check if any LLM API key is configured."""
        return bool(
            self.openai_api_key 
            or self.anthropic_api_key 
            or self.azure_openai_api_key
            or self.google_api_key
        )


# Global config instance
config = AgentLeakConfig()


def get_config() -> AgentLeakConfig:
    """Get the global configuration instance."""
    return config


def setup_logging() -> None:
    """Configure logging based on settings."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    if config.debug:
        logging.getLogger("agentleak").setLevel(logging.DEBUG)
