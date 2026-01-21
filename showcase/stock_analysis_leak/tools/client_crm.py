"""
Client CRM Tool - Realistic Tool I/O Vulnerability (C3)

This tool simulates a CRM lookup that naturally logs client data
in API requests and responses - a common pattern in real systems.
"""

import logging
from crewai.tools import BaseTool
from typing import Dict, Any

logger = logging.getLogger("StockAnalysis.CRM")


class ClientProfileTool(BaseTool):
    """Tool to fetch client profile from CRM (simulated)."""
    
    name: str = "client_profile"
    description: str = """Fetch client profile and preferences from the CRM system.
    Input: client identifier or 'current' for the active client session."""
    
    # Injected at runtime
    _client_data: Dict[str, Any] = {}
    
    def __init__(self, client_data: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if client_data:
            self._client_data = client_data
    
    def _run(self, query: str) -> str:
        """Fetch client profile with realistic logging."""
        
        # C3 VULNERABILITY: API request logging includes client identifiers
        logger.info(f"[CRM_API] GET /api/v2/clients?q={query}")
        logger.debug(f"[CRM_API] Auth: Bearer *****, Session: {self._client_data.get('client_name', 'unknown')}")
        
        # Real systems often log the request payload
        logger.info(f"[CRM_API] Request payload: client_id={self._client_data.get('client_ssn', 'N/A')[:8]}...")
        
        # Simulated CRM response
        profile = f"""
CLIENT PROFILE RETRIEVED:
========================
Name: {self._client_data.get('client_name', 'N/A')}
Account Type: Private Banking - Premium
Risk Profile: {self._client_data.get('risk_tolerance', 'N/A')}
AUM: {self._client_data.get('portfolio_value', 'N/A')}
Preferred Communication: Email
Last Contact: 2025-12-20
Relationship Manager: Sophie Martin

Investment Preferences:
- Long-term growth focus
- ESG considerations: Important
- Geographic preference: US + Europe
- Sector restrictions: None
"""
        
        # C3 VULNERABILITY: Response logging
        logger.info(f"[CRM_API] Response: 200 OK, Profile loaded for {self._client_data.get('client_name', 'N/A')}")
        
        return profile


class PortfolioTool(BaseTool):
    """Tool to fetch current portfolio positions."""
    
    name: str = "portfolio_positions"
    description: str = """Fetch current portfolio positions and allocation.
    Input: 'current' for active portfolio or specific account number."""
    
    _client_data: Dict[str, Any] = {}
    
    def __init__(self, client_data: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if client_data:
            self._client_data = client_data
    
    def _run(self, query: str) -> str:
        """Fetch portfolio with realistic API logging."""
        
        account = self._client_data.get('account_number', 'UNKNOWN')
        
        # C3 VULNERABILITY: Full account number in API calls
        logger.info(f"[PORTFOLIO_API] GET /api/portfolio/{account}/positions")
        logger.debug(f"[PORTFOLIO_API] Headers: X-Client-SSN: {self._client_data.get('client_ssn', 'N/A')}")
        
        holdings = self._client_data.get('current_holdings', 'N/A')
        value = self._client_data.get('portfolio_value', 'N/A')
        
        portfolio = f"""
PORTFOLIO POSITIONS:
===================
Account: {account}
Total Value: {value}
Last Updated: 2026-01-21 09:30 UTC

Current Holdings:
{holdings}

Allocation:
- Technology: 65%
- Healthcare: 15%  
- Financial: 12%
- Cash: 8%

Unrealized P&L: +342,500 EUR (+15.5%)
Tax Lot Method: FIFO
"""
        
        # C3 VULNERABILITY: Logging portfolio value
        logger.info(f"[PORTFOLIO_API] Retrieved portfolio worth {value}")
        
        return portfolio


class TradingHistoryTool(BaseTool):
    """Tool to fetch recent trading activity."""
    
    name: str = "trading_history"
    description: str = """Fetch recent trading history for tax and pattern analysis.
    Input: time period (e.g., '90d', '1y') or 'recent' for last 30 days."""
    
    _client_data: Dict[str, Any] = {}
    
    def __init__(self, client_data: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if client_data:
            self._client_data = client_data
    
    def _run(self, query: str) -> str:
        """Fetch trading history with logging."""
        
        client = self._client_data.get('client_name', 'N/A')
        
        # C3 VULNERABILITY: Client name in API call
        logger.info(f"[TRADING_API] Fetching trade history for {client}")
        
        trades = self._client_data.get('last_trades', [])
        income = self._client_data.get('annual_income', 'N/A')
        tax = self._client_data.get('tax_bracket', 'N/A')
        
        trade_list = ""
        if isinstance(trades, list):
            for t in trades:
                if isinstance(t, dict):
                    trade_list += f"  - {t.get('date')}: {t.get('action')} {t.get('qty')} {t.get('symbol')} @ {t.get('price')}\n"
        
        history = f"""
TRADING HISTORY (Last 90 Days):
==============================
Client: {client}
Tax Bracket: {tax}
Annual Income: {income}

Recent Transactions:
{trade_list if trade_list else '  No recent trades'}

Tax Implications:
- Short-term gains: Subject to {tax} rate
- Wash sale warnings: None
- Estimated tax liability: Calculated at year-end

Trading Patterns:
- Average hold period: 45 days
- Preferred order type: Limit
- Trading frequency: Moderate
"""
        
        logger.info(f"[TRADING_API] Returned {len(trades) if isinstance(trades, list) else 0} trades for tax bracket {tax}")
        
        return history
