#!/usr/bin/env python3
"""
AgentLeak Professional Benchmark Configuration

This module defines all models, frameworks, channels, and test configurations
for the comprehensive benchmark suite.

Configuration Categories:
- OPENROUTER_MODELS: All LLM models available via OpenRouter API
- FRAMEWORKS: Multi-agent frameworks to test (LangChain, CrewAI, AutoGPT, etc.)
- CHANNELS: The 7 leakage channels (C1-C7)
- ATTACK_LEVELS: Attack intensity levels (A0, A1, A2)
- VERTICALS: Domain verticals (healthcare, finance, legal, corporate)

Author: AgentLeak Research Team
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


# =============================================================================
# OPENROUTER MODELS CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for an LLM model on OpenRouter."""
    id: str                     # OpenRouter model ID
    name: str                   # Display name
    provider: str               # Provider (OpenAI, Anthropic, Google, etc.)
    context_length: int         # Max context in tokens
    cost_per_1k_input: float    # Cost per 1K input tokens (USD)
    cost_per_1k_output: float   # Cost per 1K output tokens (USD)
    supports_tools: bool        # Whether model supports function calling
    supports_vision: bool       # Whether model supports images
    tier: str                   # "flagship", "mid", "budget"
    

# Top-Tier Flagship Models (Best Performance)
FLAGSHIP_MODELS = {
    # OpenAI
    "gpt-4o": ModelConfig(
        id="openai/gpt-4o",
        name="GPT-4o",
        provider="OpenAI",
        context_length=128000,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        supports_tools=True,
        supports_vision=True,
        tier="flagship"
    ),
    "gpt-4-turbo": ModelConfig(
        id="openai/gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="OpenAI",
        context_length=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supports_tools=True,
        supports_vision=True,
        tier="flagship"
    ),
    "o1": ModelConfig(
        id="openai/o1",
        name="OpenAI o1",
        provider="OpenAI",
        context_length=200000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.06,
        supports_tools=False,
        supports_vision=True,
        tier="flagship"
    ),
    "o1-mini": ModelConfig(
        id="openai/o1-mini",
        name="OpenAI o1-mini",
        provider="OpenAI",
        context_length=128000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.012,
        supports_tools=False,
        supports_vision=False,
        tier="flagship"
    ),
    
    # Anthropic
    "claude-3.5-sonnet": ModelConfig(
        id="anthropic/claude-3.5-sonnet",
        name="Claude 3.5 Sonnet",
        provider="Anthropic",
        context_length=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_tools=True,
        supports_vision=True,
        tier="flagship"
    ),
    "claude-3-opus": ModelConfig(
        id="anthropic/claude-3-opus",
        name="Claude 3 Opus",
        provider="Anthropic",
        context_length=200000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        supports_tools=True,
        supports_vision=True,
        tier="flagship"
    ),
    "claude-sonnet-4": ModelConfig(
        id="anthropic/claude-sonnet-4",
        name="Claude Sonnet 4",
        provider="Anthropic",
        context_length=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_tools=True,
        supports_vision=True,
        tier="flagship"
    ),
    
    # Google
    "gemini-2.5-pro": ModelConfig(
        id="google/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="Google",
        context_length=1000000,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.01,
        supports_tools=True,
        supports_vision=True,
        tier="flagship"
    ),
    "gemini-2.0-flash": ModelConfig(
        id="google/gemini-2.0-flash-001",
        name="Gemini 2.0 Flash",
        provider="Google",
        context_length=1000000,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0004,
        supports_tools=True,
        supports_vision=True,
        tier="flagship"
    ),
}

# Mid-Tier Models (Good Balance)
MID_TIER_MODELS = {
    # OpenAI
    "gpt-4o-mini": ModelConfig(
        id="openai/gpt-4o-mini",
        name="GPT-4o Mini",
        provider="OpenAI",
        context_length=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        supports_tools=True,
        supports_vision=True,
        tier="mid"
    ),
    
    # Anthropic
    "claude-3.5-haiku": ModelConfig(
        id="anthropic/claude-3.5-haiku",
        name="Claude 3.5 Haiku",
        provider="Anthropic",
        context_length=200000,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        supports_tools=True,
        supports_vision=True,
        tier="mid"
    ),
    "claude-3-haiku": ModelConfig(
        id="anthropic/claude-3-haiku",
        name="Claude 3 Haiku",
        provider="Anthropic",
        context_length=200000,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        supports_tools=True,
        supports_vision=True,
        tier="mid"
    ),
    
    # Meta Llama
    "llama-3.3-70b": ModelConfig(
        id="meta-llama/llama-3.3-70b-instruct",
        name="Llama 3.3 70B",
        provider="Meta",
        context_length=128000,
        cost_per_1k_input=0.0004,
        cost_per_1k_output=0.0004,
        supports_tools=True,
        supports_vision=False,
        tier="mid"
    ),
    "llama-3.1-70b": ModelConfig(
        id="meta-llama/llama-3.1-70b-instruct",
        name="Llama 3.1 70B",
        provider="Meta",
        context_length=128000,
        cost_per_1k_input=0.00035,
        cost_per_1k_output=0.0004,
        supports_tools=True,
        supports_vision=False,
        tier="mid"
    ),
    "llama-3-70b": ModelConfig(
        id="meta-llama/llama-3-70b-instruct",
        name="Llama 3 70B",
        provider="Meta",
        context_length=8192,
        cost_per_1k_input=0.00035,
        cost_per_1k_output=0.0004,
        supports_tools=True,
        supports_vision=False,
        tier="mid"
    ),
    
    # Qwen
    "qwen-2.5-72b": ModelConfig(
        id="qwen/qwen-2.5-72b-instruct",
        name="Qwen 2.5 72B",
        provider="Alibaba",
        context_length=128000,
        cost_per_1k_input=0.00035,
        cost_per_1k_output=0.0004,
        supports_tools=True,
        supports_vision=False,
        tier="mid"
    ),
    
    # Mistral
    "mistral-large": ModelConfig(
        id="mistralai/mistral-large-2411",
        name="Mistral Large",
        provider="Mistral",
        context_length=128000,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
        supports_tools=True,
        supports_vision=True,
        tier="mid"
    ),
    
    # DeepSeek
    "deepseek-chat": ModelConfig(
        id="deepseek/deepseek-chat",
        name="DeepSeek Chat",
        provider="DeepSeek",
        context_length=64000,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        supports_tools=True,
        supports_vision=False,
        tier="mid"
    ),
    "deepseek-r1": ModelConfig(
        id="deepseek/deepseek-r1",
        name="DeepSeek R1",
        provider="DeepSeek",
        context_length=64000,
        cost_per_1k_input=0.00055,
        cost_per_1k_output=0.00219,
        supports_tools=True,
        supports_vision=False,
        tier="mid"
    ),
}

# Budget Models (Cost Efficient)
BUDGET_MODELS = {
    # Qwen
    "qwen-2.5-7b": ModelConfig(
        id="qwen/qwen-2.5-7b-instruct",
        name="Qwen 2.5 7B",
        provider="Alibaba",
        context_length=32000,
        cost_per_1k_input=0.00007,
        cost_per_1k_output=0.00007,
        supports_tools=True,
        supports_vision=False,
        tier="budget"
    ),
    "qwen-2.5-32b": ModelConfig(
        id="qwen/qwen-2.5-coder-32b-instruct",
        name="Qwen 2.5 Coder 32B",
        provider="Alibaba",
        context_length=32000,
        cost_per_1k_input=0.00007,
        cost_per_1k_output=0.00016,
        supports_tools=True,
        supports_vision=False,
        tier="budget"
    ),
    
    # Llama
    "llama-3.1-8b": ModelConfig(
        id="meta-llama/llama-3.1-8b-instruct",
        name="Llama 3.1 8B",
        provider="Meta",
        context_length=128000,
        cost_per_1k_input=0.00004,
        cost_per_1k_output=0.00004,
        supports_tools=True,
        supports_vision=False,
        tier="budget"
    ),
    
    # Gemini
    "gemini-flash-1.5-8b": ModelConfig(
        id="google/gemini-flash-1.5-8b",
        name="Gemini Flash 1.5 8B",
        provider="Google",
        context_length=1000000,
        cost_per_1k_input=0.0000375,
        cost_per_1k_output=0.00015,
        supports_tools=True,
        supports_vision=True,
        tier="budget"
    ),
    
    # Mistral
    "mistral-7b": ModelConfig(
        id="mistralai/mistral-7b-instruct-v0.3",
        name="Mistral 7B",
        provider="Mistral",
        context_length=32000,
        cost_per_1k_input=0.00003,
        cost_per_1k_output=0.00003,
        supports_tools=False,
        supports_vision=False,
        tier="budget"
    ),
    "mixtral-8x7b": ModelConfig(
        id="mistralai/mixtral-8x7b-instruct",
        name="Mixtral 8x7B",
        provider="Mistral",
        context_length=32000,
        cost_per_1k_input=0.00024,
        cost_per_1k_output=0.00024,
        supports_tools=True,
        supports_vision=False,
        tier="budget"
    ),
}

# All models combined
ALL_MODELS = {**FLAGSHIP_MODELS, **MID_TIER_MODELS, **BUDGET_MODELS}


# =============================================================================
# MULTI-AGENT FRAMEWORKS
# =============================================================================

class FrameworkType(Enum):
    """Supported multi-agent frameworks."""
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    AUTOGPT = "autogpt"
    METAGPT = "metagpt"
    AGENTGPT = "agentgpt"


@dataclass
class FrameworkConfig:
    """Configuration for a multi-agent framework."""
    name: str
    type: FrameworkType
    description: str
    n_agents: int
    agent_roles: List[str]
    supports_tool_use: bool
    supports_memory: bool
    supports_planning: bool
    

FRAMEWORK_CONFIGS = {
    "langchain_single": FrameworkConfig(
        name="LangChain Single Agent",
        type=FrameworkType.LANGCHAIN,
        description="Single ReAct agent with tools",
        n_agents=1,
        agent_roles=["assistant"],
        supports_tool_use=True,
        supports_memory=True,
        supports_planning=True,
    ),
    "langchain_multi": FrameworkConfig(
        name="LangChain Multi-Agent",
        type=FrameworkType.LANGCHAIN,
        description="Multi-agent graph with specialized roles",
        n_agents=3,
        agent_roles=["coordinator", "researcher", "executor"],
        supports_tool_use=True,
        supports_memory=True,
        supports_planning=True,
    ),
    "crewai": FrameworkConfig(
        name="CrewAI",
        type=FrameworkType.CREWAI,
        description="Role-based collaborative agents",
        n_agents=3,
        agent_roles=["manager", "analyst", "executor"],
        supports_tool_use=True,
        supports_memory=True,
        supports_planning=True,
    ),
    "autogpt": FrameworkConfig(
        name="AutoGPT",
        type=FrameworkType.AUTOGPT,
        description="Autonomous goal-driven agent",
        n_agents=1,
        agent_roles=["autonomous_agent"],
        supports_tool_use=True,
        supports_memory=True,
        supports_planning=True,
    ),
    "metagpt": FrameworkConfig(
        name="MetaGPT",
        type=FrameworkType.METAGPT,
        description="Software company simulation",
        n_agents=5,
        agent_roles=["product_manager", "architect", "developer", "qa", "reviewer"],
        supports_tool_use=True,
        supports_memory=True,
        supports_planning=True,
    ),
}


# =============================================================================
# LEAKAGE CHANNELS
# =============================================================================

class Channel(Enum):
    """The 7 leakage channels in AgentLeak."""
    C1_FINAL_OUTPUT = "C1_final_output"
    C2_INTER_AGENT = "C2_inter_agent"
    C3_TOOL_INPUT = "C3_tool_input"
    C4_TOOL_OUTPUT = "C4_tool_output"
    C5_MEMORY = "C5_memory"
    C6_LOGS = "C6_logs"
    C7_ARTIFACTS = "C7_artifacts"


CHANNEL_INFO = {
    Channel.C1_FINAL_OUTPUT: {
        "name": "Final Output",
        "description": "User-visible response from the agent",
        "risk_level": "HIGH",
        "visibility": "direct",
    },
    Channel.C2_INTER_AGENT: {
        "name": "Inter-Agent Communication",
        "description": "Messages between agents in multi-agent systems",
        "risk_level": "HIGH",
        "visibility": "internal",
    },
    Channel.C3_TOOL_INPUT: {
        "name": "Tool Input",
        "description": "Arguments passed to external tool calls",
        "risk_level": "MEDIUM",
        "visibility": "logged",
    },
    Channel.C4_TOOL_OUTPUT: {
        "name": "Tool Output",
        "description": "Results returned from tool execution",
        "risk_level": "MEDIUM",
        "visibility": "logged",
    },
    Channel.C5_MEMORY: {
        "name": "Agent Memory",
        "description": "Persistent memory stores and embeddings",
        "risk_level": "MEDIUM",
        "visibility": "persistent",
    },
    Channel.C6_LOGS: {
        "name": "Debug Logs",
        "description": "System logs and debug traces",
        "risk_level": "LOW",
        "visibility": "logged",
    },
    Channel.C7_ARTIFACTS: {
        "name": "Artifacts",
        "description": "Generated files, code, and documents",
        "risk_level": "HIGH",
        "visibility": "persisted",
    },
}


# =============================================================================
# ATTACK LEVELS
# =============================================================================

class AttackLevel(Enum):
    """Attack intensity levels."""
    A0_BENIGN = "A0"       # No attack, benign user
    A1_INDIRECT = "A1"     # Indirect prompt injection
    A2_ADVERSARIAL = "A2"  # Adversarial multi-turn attacks


ATTACK_INFO = {
    AttackLevel.A0_BENIGN: {
        "name": "Benign",
        "description": "No attack - legitimate user request",
        "techniques": ["none"],
    },
    AttackLevel.A1_INDIRECT: {
        "name": "Indirect Prompt Injection",
        "description": "Hidden instructions in tool outputs or data",
        "techniques": [
            "tool_output_poisoning",
            "data_exfiltration",
            "memory_manipulation",
        ],
    },
    AttackLevel.A2_ADVERSARIAL: {
        "name": "Adversarial Multi-Turn",
        "description": "Sophisticated multi-step attacks",
        "techniques": [
            "jailbreak_chain",
            "role_confusion",
            "context_manipulation",
            "cross_agent_collusion",
            "privilege_escalation",
        ],
    },
}


# =============================================================================
# DOMAIN VERTICALS
# =============================================================================

class Vertical(Enum):
    """Domain verticals for scenarios."""
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    LEGAL = "legal"
    CORPORATE = "corporate"


VERTICAL_INFO = {
    Vertical.HEALTHCARE: {
        "sensitive_fields": ["ssn", "medical_record_number", "diagnosis", "treatment"],
        "regulations": ["HIPAA", "GDPR"],
    },
    Vertical.FINANCE: {
        "sensitive_fields": ["account_number", "credit_card", "balance", "transactions"],
        "regulations": ["PCI-DSS", "SOX", "GDPR"],
    },
    Vertical.LEGAL: {
        "sensitive_fields": ["case_number", "client_communications", "legal_strategy"],
        "regulations": ["Attorney-Client Privilege", "GDPR"],
    },
    Vertical.CORPORATE: {
        "sensitive_fields": ["employee_id", "salary", "performance_review", "trade_secrets"],
        "regulations": ["GDPR", "CCPA", "NDAs"],
    },
}


# =============================================================================
# BENCHMARK PROFILES
# =============================================================================

@dataclass
class BenchmarkProfile:
    """Pre-configured benchmark profile."""
    name: str
    description: str
    models: List[str]
    frameworks: List[str]
    channels: List[Channel]
    attack_levels: List[AttackLevel]
    n_scenarios: int
    max_cost_usd: float
    

# Predefined profiles
BENCHMARK_PROFILES = {
    "quick": BenchmarkProfile(
        name="Quick Test",
        description="Fast validation run with budget models",
        models=["gpt-4o-mini", "llama-3.1-8b"],
        frameworks=["langchain_single"],
        channels=[Channel.C1_FINAL_OUTPUT, Channel.C3_TOOL_INPUT],
        attack_levels=[AttackLevel.A0_BENIGN],
        n_scenarios=10,
        max_cost_usd=0.10,
    ),
    "standard": BenchmarkProfile(
        name="Standard Benchmark",
        description="Balanced coverage across models and scenarios",
        models=["gpt-4o-mini", "claude-3.5-haiku", "qwen-2.5-72b", "llama-3.3-70b"],
        frameworks=["langchain_single", "langchain_multi"],
        channels=list(Channel),
        attack_levels=[AttackLevel.A0_BENIGN, AttackLevel.A1_INDIRECT],
        n_scenarios=100,
        max_cost_usd=5.00,
    ),
    "comprehensive": BenchmarkProfile(
        name="Comprehensive Benchmark",
        description="Full benchmark across all models, frameworks, and attacks",
        models=list(ALL_MODELS.keys()),
        frameworks=list(FRAMEWORK_CONFIGS.keys()),
        channels=list(Channel),
        attack_levels=list(AttackLevel),
        n_scenarios=500,
        max_cost_usd=100.00,
    ),
    "flagship": BenchmarkProfile(
        name="Flagship Models",
        description="Test only flagship models for paper results",
        models=list(FLAGSHIP_MODELS.keys()),
        frameworks=["langchain_single", "langchain_multi", "crewai"],
        channels=list(Channel),
        attack_levels=list(AttackLevel),
        n_scenarios=200,
        max_cost_usd=50.00,
    ),
    "paper": BenchmarkProfile(
        name="Paper Results",
        description="Full benchmark for paper publication",
        models=[
            "gpt-4o",
            "gemini-2.0-flash",
            "claude-3-haiku",
            "qwen-2.5-72b",
            "gemini-2.5-pro",
            "llama-3-70b",
        ],
        frameworks=["langchain_single", "langchain_multi", "crewai", "autogpt"],
        channels=list(Channel),
        attack_levels=list(AttackLevel),
        n_scenarios=1000,
        max_cost_usd=200.00,
    ),
}


# =============================================================================
# COST ESTIMATION
# =============================================================================

def estimate_benchmark_cost(
    profile: BenchmarkProfile,
    avg_input_tokens: int = 2000,
    avg_output_tokens: int = 500,
) -> Dict[str, Any]:
    """
    Estimate the cost of running a benchmark profile.
    
    Returns:
        Dict with cost breakdown by model and total.
    """
    cost_breakdown = {}
    total_cost = 0.0
    
    n_tests_per_model = (
        profile.n_scenarios 
        * len(profile.frameworks) 
        * len(profile.attack_levels)
    )
    
    for model_name in profile.models:
        if model_name not in ALL_MODELS:
            continue
            
        model = ALL_MODELS[model_name]
        input_cost = (avg_input_tokens / 1000) * model.cost_per_1k_input * n_tests_per_model
        output_cost = (avg_output_tokens / 1000) * model.cost_per_1k_output * n_tests_per_model
        model_cost = input_cost + output_cost
        
        cost_breakdown[model_name] = {
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total": round(model_cost, 4),
        }
        total_cost += model_cost
    
    return {
        "models": cost_breakdown,
        "total_estimated_cost": round(total_cost, 2),
        "n_models": len(profile.models),
        "n_tests_per_model": n_tests_per_model,
        "total_api_calls": n_tests_per_model * len(profile.models),
        "within_budget": total_cost <= profile.max_cost_usd,
    }


def get_model_by_name(name: str) -> Optional[ModelConfig]:
    """Get model config by name."""
    return ALL_MODELS.get(name)


def list_models_by_tier(tier: str) -> List[str]:
    """List models by tier (flagship, mid, budget)."""
    return [
        name for name, config in ALL_MODELS.items()
        if config.tier == tier
    ]


def list_models_with_tools() -> List[str]:
    """List models that support function calling."""
    return [
        name for name, config in ALL_MODELS.items()
        if config.supports_tools
    ]


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("AgentLeak Benchmark Configuration Summary")
    print("=" * 60)
    
    print(f"\n📊 Models Available: {len(ALL_MODELS)}")
    print(f"  - Flagship: {len(FLAGSHIP_MODELS)}")
    print(f"  - Mid-tier: {len(MID_TIER_MODELS)}")
    print(f"  - Budget: {len(BUDGET_MODELS)}")
    
    print(f"\n🤖 Frameworks: {len(FRAMEWORK_CONFIGS)}")
    for name, config in FRAMEWORK_CONFIGS.items():
        print(f"  - {config.name}: {config.n_agents} agents")
    
    print(f"\n📡 Channels: {len(Channel)}")
    for ch in Channel:
        info = CHANNEL_INFO[ch]
        print(f"  - {info['name']}: {info['risk_level']} risk")
    
    print(f"\n⚔️ Attack Levels: {len(AttackLevel)}")
    for al in AttackLevel:
        print(f"  - {al.value}: {ATTACK_INFO[al]['name']}")
    
    print(f"\n📦 Benchmark Profiles:")
    for name, profile in BENCHMARK_PROFILES.items():
        estimate = estimate_benchmark_cost(profile)
        print(f"  - {profile.name}: {profile.n_scenarios} scenarios, "
              f"~${estimate['total_estimated_cost']:.2f}")
