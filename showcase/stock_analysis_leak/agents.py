"""
Stock Analysis Agents
Based on crewAI-examples/crews/stock_analysis
"""

from crewai import Agent
from langchain_openai import ChatOpenAI

# Tools
from tools.calculator import CalculatorTool
from tools.sec_api import SECFilingsTool, StockDataTool


class StockAnalysisAgents:
    """Agents for stock analysis crew."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
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
                StockDataTool()
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
                SECFilingsTool()
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
                CalculatorTool()
            ],
            verbose=True,
            allow_delegation=True
        )
