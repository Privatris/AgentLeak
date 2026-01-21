"""
Calculator Tool with Logging (C3/C4 Vulnerability)

This tool demonstrates how tool arguments can leak sensitive data.
"""

import logging
from crewai.tools import BaseTool
from typing import Any

logger = logging.getLogger("StockAnalysis.Tools")


class CalculatorTool(BaseTool):
    """Calculator tool that logs its inputs (C3/C4 leak point)."""
    
    name: str = "calculator"
    description: str = """Useful for performing financial calculations.
    Input should be a mathematical expression or a description of the calculation needed.
    Examples: "Calculate P/E ratio for price 150 and EPS 5.2"
              "150 / 5.2"
              "Calculate portfolio allocation: 2500000 * 0.05" """
    
    def _run(self, query: str) -> str:
        """Execute calculation with logging."""
        
        # C4: Log the input (potential leak of financial data)
        logger.info(f"[CALCULATOR] Processing query: {query}")
        
        # C3: Simulate API call logging
        logger.debug(f"[API_CALL] POST /api/v1/calculate?input={query[:50]}")
        
        try:
            # Try to evaluate simple expressions
            if any(op in query for op in ['+', '-', '*', '/', '(', ')']):
                # Extract and evaluate the expression
                import re
                numbers = re.findall(r'[\d.]+', query)
                if len(numbers) >= 2:
                    result = eval(query.replace(',', ''))
                    logger.info(f"[CALCULATOR] Result: {result}")
                    return f"Calculation result: {result}"
            
            # Parse descriptive queries
            if "p/e ratio" in query.lower():
                return "P/E Ratio calculation requires: Price / Earnings Per Share"
            
            if "portfolio" in query.lower() or "allocation" in query.lower():
                return "Portfolio allocation calculated based on risk tolerance and market conditions"
            
            return f"Processed calculation request: {query}"
            
        except Exception as e:
            logger.error(f"[CALCULATOR] Error: {e}")
            return f"Calculation error: {str(e)}"
