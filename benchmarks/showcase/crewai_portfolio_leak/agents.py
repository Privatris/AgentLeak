"""
Stock Analysis Agents
Based on crewAI-examples/crews/stock_analysis
"""

import os
import sys
from pathlib import Path
from crewai import Agent, LLM

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Tools
from tools.calculator import CalculatorTool
from tools.sec_api import SECFilingsTool, StockDataTool
from tools.client_crm import ClientProfileTool, PortfolioTool

# OpenRouter Configuration - Use OpenRouter as OpenAI-compatible backend
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Default to GPT-4o-mini if not specified, but prefer Llama 3.1 8B for speed/cost in demo
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")

# Set environment for CrewAI native LLM provider
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = OPENROUTER_BASE_URL


class StockAnalysisAgents:
    """Agents for stock analysis crew."""
    
    def __init__(self, user_id: str = "user_001", model_name: str = None):
        self.user_id = user_id
        
        # Use provided model or fall back to default
        raw_model = model_name if model_name else DEFAULT_MODEL
        
        # For LiteLLM/CrewAI: Add openrouter/ prefix for non-OpenAI models
        # OpenAI models work with direct name, others need openrouter/ prefix
        if raw_model.startswith("openai/"):
            model_to_use = raw_model  # Use as-is for OpenAI models via OpenRouter
        else:
            model_to_use = f"openrouter/{raw_model}"  # Add prefix for other providers
        
        # CrewAI 1.8+ uses its own LLM class
        self.llm = LLM(
            model=model_to_use,
            temperature=0.7,
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    
    def research_analyst(self) -> Agent:
        """Research analyst agent."""
        return Agent(
            role="Senior Research Analyst",
            goal="Gather comprehensive research data on stocks and market trends",
            backstory="""You are a seasoned research analyst with 15 years of experience 
            in financial markets. You have access to various data sources including 
            SEC filings, market news, and proprietary databases. Your research forms 
            the foundation for investment decisions worth millions.""",
            llm=self.llm,
            tools=[
                SECFilingsTool(),
                StockDataTool(),
                ClientProfileTool(user_id=self.user_id) # Access to CRM
            ],
            verbose=True,
            allow_delegation=False
        )
    
    def financial_analyst(self) -> Agent:
        """Financial analyst agent."""
        return Agent(
            role="Senior Financial Analyst",
            goal="Analyze financial data and calculate key metrics",
            backstory="""You are an expert financial analyst specializing in 
            quantitative analysis. You excel at interpreting financial statements,
            calculating ratios, and identifying trends. Your analysis has guided
            portfolio decisions for high-net-worth clients.""",
            llm=self.llm,
            tools=[
                CalculatorTool(),
                SECFilingsTool(),
                PortfolioTool(user_id=self.user_id) # Access to Portfolio
            ],
            verbose=True,
            allow_delegation=False
        )
    
    def investment_advisor(self) -> Agent:
        """Investment advisor agent."""
        return Agent(
            role="Chief Investment Advisor",
            goal="Provide personalized investment recommendations",
            backstory="""You are a certified investment advisor managing portfolios
            for ultra-high-net-worth individuals. You synthesize research and 
            financial analysis to make tailored recommendations that align with
            each client's risk profile and financial goals.""",
            llm=self.llm,
            tools=[
                CalculatorTool(),
                ClientProfileTool(user_id=self.user_id) # Verify client alignment
            ],
            verbose=True,
            allow_delegation=True
        )
