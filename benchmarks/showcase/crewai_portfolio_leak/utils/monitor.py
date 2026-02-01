from typing import Dict, List, Optional
from datetime import datetime
import logging

try:
    from agentleak import AgentLeakTester, DetectionMode
except ImportError:
    # Use mock if SDK not installed (should not happen in this env)
    class AgentLeakTester:
        def __init__(self, mode): pass
        def check(self, vault, output, channel): 
            class Result:
                leaked = False
            return Result()
    class DetectionMode:
        FAST = "fast"
        ACCURATE = "accurate"
        HYBRID = "hybrid"

class SDKChannelMonitor:
    """Uses AgentLeakTester SDK to monitor multiple channels."""
    
    def __init__(self, vault: Dict[str, str], mode: DetectionMode = DetectionMode.HYBRID):
        self.vault = vault
        self.tester = AgentLeakTester(mode=mode)
        self.detections: List[Dict] = []
        
    def check(self, content: str, channel: str, source: str) -> Optional[Dict]:
        """Check content for leaks using SDK."""
        if not content:
            return None
            
        result = self.tester.check(
            vault=self.vault,
            output=content,
            channel=channel
        )
        
        if result and result.leaked:
            detection = {
                "channel": channel,
                "source": source,
                "leaked_items": result.detected_items,
                "confidence": result.confidence,
                "tier": result.tier_used,
                "explanation": getattr(result, "explanation", "No explanation"),
                "timestamp": datetime.now().isoformat(),
            }
            self.detections.append(detection)
            return detection
        return None
    
    def get_summary(self) -> Dict[str, int]:
        """Get leak count by channel."""
        summary = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "C6": 0}
        for d in self.detections:
            ch = d["channel"]
            if ch in summary:
                summary[ch] += 1
        return summary
    
    def get_stats(self) -> Dict:
        """Get detailed statistics."""
        return {
            "total_leaks": len(self.detections),
            "by_channel": self.get_summary(),
            "detections": self.detections,
        }
