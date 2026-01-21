"""
AgentLeak Leakage Channels - 7 channels (C1-C7) from the paper.

From the paper "AgentLeak: Privacy Benchmark for Multi-Agent LLM Systems":

    C1: Final Output (user-visible response)
    C2: Inter-Agent Messages (agent-to-agent communication)
    C3: Tool Inputs (arguments passed to tools)
    C4: Tool Outputs (results returned from tools)
    C5: Memory Writes (persistent storage)
    C6: Logs (debug/audit logs)
    C7: Artifacts (generated files, images, exports)

Key insight: Prior benchmarks only audit C1. AgentLeak audits all 7 channels.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union


class Channel(str, Enum):
    """The 7 leakage channels from the paper."""
    C1 = "C1"  # Final Output
    C2 = "C2"  # Inter-Agent Messages
    C3 = "C3"  # Tool Inputs
    C4 = "C4"  # Tool Outputs
    C5 = "C5"  # Memory Writes
    C6 = "C6"  # Logs
    C7 = "C7"  # Artifacts

    @property
    def name(self) -> str:
        return {
            "C1": "Final Output",
            "C2": "Inter-Agent Messages",
            "C3": "Tool Inputs",
            "C4": "Tool Outputs",
            "C5": "Memory Writes",
            "C6": "Logs",
            "C7": "Artifacts",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            "C1": "User-visible response from the agent",
            "C2": "Messages exchanged between agents in multi-agent systems",
            "C3": "Arguments and parameters passed to external tools",
            "C4": "Results returned from tool executions",
            "C5": "Data written to persistent memory/storage",
            "C6": "Debug, audit, and system logs",
            "C7": "Generated files, images, reports, and exports",
        }[self.value]

    @property
    def visibility(self) -> str:
        """Who can observe this channel."""
        return {
            "C1": "user",          # User sees final output
            "C2": "internal",      # Only agents see inter-agent messages
            "C3": "tool_provider", # Tool providers see inputs
            "C4": "internal",      # Agents see tool outputs
            "C5": "persistent",    # Anyone with storage access
            "C6": "admin",         # Admins/developers see logs
            "C7": "user",          # Users receive artifacts
        }[self.value]

    @property
    def prior_coverage(self) -> bool:
        """Whether prior benchmarks covered this channel."""
        # Only C1 was covered by prior work
        return self.value == "C1"


@dataclass
class ChannelInfo:
    """Detailed information about a leakage channel."""
    channel: Channel
    detection_methods: List[str]
    typical_leakage_patterns: List[str]
    defense_strategies: List[str]
    risk_level: str  # "high", "medium", "low"


# Detailed channel information
CHANNEL_INFO = {
    Channel.C1: ChannelInfo(
        channel=Channel.C1,
        detection_methods=["regex_matching", "semantic_similarity", "canary_detection", "llm_judge"],
        typical_leakage_patterns=["direct_pii_output", "formatted_data_tables", "json_dumps"],
        defense_strategies=["output_sanitization", "pii_redaction", "response_filtering"],
        risk_level="high",
    ),
    Channel.C2: ChannelInfo(
        channel=Channel.C2,
        detection_methods=["message_interception", "semantic_analysis", "pattern_matching"],
        typical_leakage_patterns=["data_passing", "context_sharing", "delegation_leaks"],
        defense_strategies=["message_encryption", "role_isolation", "access_control"],
        risk_level="high",
    ),
    Channel.C3: ChannelInfo(
        channel=Channel.C3,
        detection_methods=["argument_inspection", "api_monitoring", "parameter_validation"],
        typical_leakage_patterns=["pii_in_queries", "sensitive_params", "exfiltration_urls"],
        defense_strategies=["input_validation", "parameter_sanitization", "url_allowlisting"],
        risk_level="medium",
    ),
    Channel.C4: ChannelInfo(
        channel=Channel.C4,
        detection_methods=["output_scanning", "content_analysis", "injection_detection"],
        typical_leakage_patterns=["poisoned_results", "injected_instructions", "data_exposure"],
        defense_strategies=["output_validation", "content_filtering", "sandboxing"],
        risk_level="medium",
    ),
    Channel.C5: ChannelInfo(
        channel=Channel.C5,
        detection_methods=["memory_auditing", "write_monitoring", "persistence_tracking"],
        typical_leakage_patterns=["pii_storage", "cross_session_leaks", "memory_poisoning"],
        defense_strategies=["memory_encryption", "access_control", "ttl_policies"],
        risk_level="high",
    ),
    Channel.C6: ChannelInfo(
        channel=Channel.C6,
        detection_methods=["log_scanning", "pattern_matching", "sensitive_data_detection"],
        typical_leakage_patterns=["pii_in_logs", "debug_dumps", "error_messages"],
        defense_strategies=["log_redaction", "structured_logging", "access_restriction"],
        risk_level="medium",
    ),
    Channel.C7: ChannelInfo(
        channel=Channel.C7,
        detection_methods=["file_scanning", "content_extraction", "metadata_analysis"],
        typical_leakage_patterns=["pii_in_reports", "data_exports", "embedded_metadata"],
        defense_strategies=["artifact_sanitization", "export_controls", "watermarking"],
        risk_level="high",
    ),
}


class ChannelRegistry:
    """Central registry of all channels."""
    
    def __init__(self):
        self._channels = {c.value: c for c in Channel}
        self._info = CHANNEL_INFO
    
    def get(self, id: str) -> Optional[Channel]:
        """Get channel by ID (e.g., 'C1')."""
        return self._channels.get(id.upper())
    
    def info(self, channel: Union[str, Channel]) -> Optional[ChannelInfo]:
        """Get detailed info for a channel."""
        if isinstance(channel, str):
            channel = self.get(channel)
        return self._info.get(channel)
    
    def all(self) -> List[Channel]:
        """Get all channels."""
        return list(Channel)
    
    def internal_only(self) -> List[Channel]:
        """Get channels that are internal (not user-visible)."""
        return [c for c in Channel if c.visibility in ["internal", "admin"]]
    
    def user_visible(self) -> List[Channel]:
        """Get channels that are user-visible."""
        return [c for c in Channel if c.visibility == "user"]
    
    def high_risk(self) -> List[Channel]:
        """Get high-risk channels."""
        return [c for c, info in self._info.items() if info.risk_level == "high"]
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_channels": 7,
            "prior_coverage": 1,  # Only C1
            "new_coverage": 6,    # C2-C7
            "by_visibility": {
                "user": len(self.user_visible()),
                "internal": len(self.internal_only()),
            },
            "by_risk": {
                "high": len(self.high_risk()),
                "medium": 7 - len(self.high_risk()),
            },
        }


# Global registry instance
CHANNELS = ChannelRegistry()
