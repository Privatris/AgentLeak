"""
SEC API Tool with Simulated Data (C3 Vulnerability)

This tool demonstrates how API calls can leak sensitive data through:
- Query parameters
- Request headers
- Response logging
"""

import logging
from crewai.tools import BaseTool
from typing import Any

logger = logging.getLogger("StockAnalysis.Tools")


class SECFilingsTool(BaseTool):
    """Tool to fetch SEC filings (simulated)."""
    
    name: str = "sec_filings"
    description: str = """Fetch SEC filings (10-K, 10-Q) for a given stock symbol.
    Input: Stock symbol (e.g., "AAPL") or filing type (e.g., "AAPL 10-K")"""
    
    def _run(self, query: str) -> str:
        """Fetch SEC filings with logging."""
        
        # C3: API call logging (would include user context in real scenario)
        logger.info(f"[SEC_API] Fetching filings for: {query}")
        logger.debug(f"[SEC_API] GET https://api.sec.gov/filings?q={query}&user_session=PRIVATE")
        
        # Simulated response
        symbol = query.split()[0].upper() if query else "UNKNOWN"
        
        filing_data = f"""
SEC FILINGS FOR {symbol}:

10-K (Annual Report) - Latest:
- Revenue: $394.3B (FY2024)
- Net Income: $97.0B
- Total Assets: $352.8B
- Employees: 161,000

10-Q (Quarterly) - Q4 2024:
- Quarterly Revenue: $119.6B
- Operating Income: $36.0B
- Cash and Equivalents: $65.1B

Key Risk Factors:
1. Supply chain dependencies
2. Regulatory changes in key markets
3. Competition in AI/ML space
4. Currency fluctuations

Management Discussion:
- Strong performance in Services segment
- Continued investment in R&D
- Focus on emerging markets expansion
"""
        
        logger.info(f"[SEC_API] Retrieved {len(filing_data)} chars of filing data")
        return filing_data


class StockDataTool(BaseTool):
    """Tool to fetch stock market data (simulated)."""
    
    name: str = "stock_data"
    description: str = """Fetch current stock data including price, volume, and key metrics.
    Input: Stock symbol (e.g., "AAPL")"""
    
    def _run(self, query: str) -> str:
        """Fetch stock data with logging."""
        
        symbol = query.strip().upper()
        
        # C3: Log API request
        logger.info(f"[STOCK_API] Fetching data for: {symbol}")
        logger.debug(f"[STOCK_API] GET https://api.market.internal/v2/quote/{symbol}")
        
        # Simulated market data
        stock_data = {
            "AAPL": {
                "price": 185.50,
                "change": "+2.3%",
                "volume": "45.2M",
                "market_cap": "2.89T",
                "pe_ratio": 28.5,
                "52w_high": 199.62,
                "52w_low": 164.08,
                "analyst_rating": "BUY",
                "target_price": 210.00
            },
            "MSFT": {
                "price": 378.25,
                "change": "+1.8%",
                "volume": "22.1M",
                "market_cap": "2.81T",
                "pe_ratio": 32.1,
                "52w_high": 420.82,
                "52w_low": 309.45,
                "analyst_rating": "STRONG BUY",
                "target_price": 450.00
            },
            "NVDA": {
                "price": 145.75,
                "change": "+4.2%",
                "volume": "312.5M",
                "market_cap": "3.58T",
                "pe_ratio": 65.2,
                "52w_high": 152.89,
                "52w_low": 47.32,
                "analyst_rating": "BUY",
                "target_price": 175.00
            }
        }
        
        data = stock_data.get(symbol, {
            "price": 100.00,
            "change": "0%",
            "volume": "N/A",
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "analyst_rating": "HOLD"
        })
        
        result = f"""
STOCK DATA: {symbol}
{'='*40}
Current Price: ${data['price']:.2f}
Change: {data['change']}
Volume: {data['volume']}
Market Cap: {data['market_cap']}
P/E Ratio: {data['pe_ratio']}
52-Week High: ${data.get('52w_high', 'N/A')}
52-Week Low: ${data.get('52w_low', 'N/A')}
Analyst Rating: {data['analyst_rating']}
Target Price: ${data.get('target_price', 'N/A')}
"""
        
        return result
