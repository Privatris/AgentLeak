"""
Report Artifact Generator - C6 Vulnerability (File Artifacts)

This module generates investment reports that are saved as files,
demonstrating how artifacts can contain leaked PII.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("StockAnalysis.Reports")


class ReportGenerator:
    """Generates investment reports with client data (C6 vulnerability)."""
    
    def __init__(self, client_data: Dict[str, Any], output_dir: Path = None):
        self.client_data = client_data
        self.output_dir = output_dir or Path(__file__).parent.parent / "generated_reports"
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_analysis_report(self, stock: str, analysis: str) -> Path:
        """Generate analysis report file (C6 vulnerability)."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{stock}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # C6 VULNERABILITY: Client data embedded in artifact
        report = {
            "report_type": "Investment Analysis",
            "generated_at": datetime.now().isoformat(),
            "stock_symbol": stock,
            
            # PII leaked into artifact
            "client_context": {
                "name": self.client_data.get("client_name"),
                "portfolio_value": self.client_data.get("portfolio_value"),
                "risk_profile": self.client_data.get("risk_tolerance"),
            },
            
            "analysis_summary": analysis[:500] if analysis else "No analysis provided",
            
            "metadata": {
                "analyst_system": "CrewAI Multi-Agent",
                "confidence": 0.85,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[REPORT] Generated: {filepath}")
        return filepath
    
    def generate_client_memo(self, stock: str, recommendation: str) -> Path:
        """Generate client-facing memo (more PII exposure)."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"client_memo_{stock}_{timestamp}.md"
        filepath = self.output_dir / filename
        
        # C6 VULNERABILITY: Full client details in memo
        memo = f"""# Investment Memo - {stock}

**Prepared for:** {self.client_data.get('client_name')}  
**Account:** {self.client_data.get('account_number')}  
**Date:** {datetime.now().strftime('%Y-%m-%d')}

---

## Executive Summary

Based on our analysis, we provide the following recommendation for {stock}
tailored to your investment profile.

## Client Profile Summary

| Attribute | Value |
|-----------|-------|
| Risk Tolerance | {self.client_data.get('risk_tolerance')} |
| Portfolio Value | {self.client_data.get('portfolio_value')} |
| Annual Income | {self.client_data.get('annual_income')} |
| Current Holdings | {self.client_data.get('current_holdings')} |

## Recommendation

{recommendation}

---

*This memo contains confidential client information and is intended solely 
for {self.client_data.get('client_name')}. Do not distribute.*
"""
        
        with open(filepath, 'w') as f:
            f.write(memo)
        
        logger.info(f"[MEMO] Generated: {filepath}")
        return filepath
    
    def generate_trade_log(self, stock: str, action: str, rationale: str) -> Path:
        """Generate trade log with audit trail (C4+C6 combined)."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_log_{timestamp}.jsonl"
        filepath = self.output_dir / filename
        
        # C6 VULNERABILITY: SSN and account in trade log
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": "RECOMMENDATION",
            "stock": stock,
            "suggested_action": action,
            "client": {
                "id": self.client_data.get("client_ssn"),
                "name": self.client_data.get("client_name"),
                "account": self.client_data.get("account_number"),
            },
            "portfolio_snapshot": {
                "value": self.client_data.get("portfolio_value"),
                "holdings": self.client_data.get("current_holdings"),
            },
            "rationale": rationale[:200] if rationale else "",
            "system": "AgentLeak-Showcase"
        }
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        
        logger.info(f"[TRADE_LOG] Entry added: {filepath}")
        return filepath


class MemoryStore:
    """Simulated memory store for agent state (C5 vulnerability)."""
    
    def __init__(self, storage_file: Path = None):
        self.storage_file = storage_file or Path(__file__).parent.parent / "agent_memory.json"
        self.memory = {}
        self._load()
    
    def _load(self):
        """Load existing memory."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file) as f:
                    self.memory = json.load(f)
            except:
                self.memory = {}
    
    def _save(self):
        """Persist memory to disk (C5+C6 vulnerability)."""
        with open(self.storage_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def store_client_context(self, client_data: Dict[str, Any]):
        """Store client context in memory (C5 vulnerability)."""
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M")
        
        # C5 VULNERABILITY: PII persisted in memory
        self.memory[session_id] = {
            "client": client_data.get("client_name"),
            "ssn_hash": client_data.get("client_ssn")[:8] + "...",  # Partial but still PII
            "portfolio": client_data.get("portfolio_value"),
            "risk": client_data.get("risk_tolerance"),
            "holdings": client_data.get("current_holdings"),
            "stored_at": datetime.now().isoformat(),
        }
        
        self._save()
        logger.info(f"[MEMORY] Stored context for session {session_id}")
        
        return session_id
    
    def store_interaction(self, agent_name: str, content: str, client_name: str = None):
        """Store agent interaction (C5 vulnerability)."""
        
        key = f"interaction_{datetime.now().strftime('%H%M%S')}"
        
        # C5 VULNERABILITY: Agent outputs with client context persisted
        self.memory[key] = {
            "agent": agent_name,
            "client": client_name,
            "content_preview": content[:300] if content else "",
            "timestamp": datetime.now().isoformat(),
        }
        
        self._save()
        return key
    
    def get_recent_clients(self) -> list:
        """Retrieve recent client contexts (demonstrates persistence risk)."""
        clients = []
        for k, v in self.memory.items():
            if "client" in v and v["client"]:
                clients.append(v["client"])
        return list(set(clients))
