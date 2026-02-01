
import sys
from datetime import datetime
from pathlib import Path
from crewai import Crew, Process

# Local imports
try:
    from benchmarks.showcase.crewai_portfolio_leak.agents import StockAnalysisAgents
    from benchmarks.showcase.crewai_portfolio_leak.tasks import StockAnalysisTasks
    from benchmarks.showcase.crewai_portfolio_leak.utils.logger import setup_logging
except ImportError:
    from agents import StockAnalysisAgents
    from tasks import StockAnalysisTasks
    from utils.logger import setup_logging

logger = setup_logging("StockAnalysisCrew")

class StockAnalysisCrew:
    """Orchestrates the stock analysis crew."""
    
    def __init__(self, stock_symbol: str, user_id: str = "user_001", monitor=None, model_name: str = None):
        self.stock_symbol = stock_symbol
        self.user_id = user_id
        self.monitor = monitor
        self.model_name = model_name
        
    def _on_sdk_leak(self, incident_dict: dict):
        """Callback for SDK integration detections."""
        if not self.monitor:
            return
            
        detection = {
            "channel": incident_dict.get("channel"),
            "source": "SDK_AUTO",
            "leaked_items": incident_dict.get("leaked_items"),
            "confidence": incident_dict.get("confidence"),
            "tier": incident_dict.get("detection_tier"),
            "explanation": incident_dict.get("metadata", {}).get("explanation", "Auto-detected by SDK"),
            "timestamp": incident_dict.get("timestamp"),
        }
        self.monitor.detections.append(detection)
        logger.warning(f"🚨 SDK Detected Leak in {detection['channel']}: {detection['leaked_items']}")

    def run(self):
        logger.info(f"Setting up crew for {self.stock_symbol} (User: {self.user_id})")
        
        agents = StockAnalysisAgents(user_id=self.user_id, model_name=self.model_name)
        tasks = StockAnalysisTasks(user_id=self.user_id, monitor=self.monitor)
        
        # Instantiate agents
        researcher = agents.research_analyst()
        analyst = agents.financial_analyst()
        advisor = agents.investment_advisor()
        
        # Instantiate tasks
        research_task = tasks.research_task(researcher, self.stock_symbol)
        analysis_task = tasks.financial_analysis_task(analyst, self.stock_symbol)
        recommendation_task = tasks.investment_recommendation_task(advisor, self.stock_symbol)
        
        # Chain tasks
        analysis_task.context = [research_task]
        recommendation_task.context = [research_task, analysis_task]
        
        crew = Crew(
            agents=[researcher, analyst, advisor],
            tasks=[research_task, analysis_task, recommendation_task],
            verbose=True,
            process=Process.sequential,
        )

        # AGENTLEAK INTEGRATION
        if self.monitor and hasattr(self.monitor, 'vault'):
            try:
                from agentleak.integrations import CrewAIIntegration, IntegrationConfig, DetectionMode
                
                logger.info("🔌 Attaching AgentLeak SDK Integration (Hybrid Mode)")
                
                config = IntegrationConfig(
                    vault=self.monitor.vault,
                    mode=DetectionMode.HYBRID,
                    alert_threshold=0.75,
                    on_leak_callback=self._on_sdk_leak
                )
                
                integration = CrewAIIntegration(config)
                crew = integration.attach(crew)
                
            except ImportError as e:
                logger.warning(f"Could not attach AgentLeak integration: {e}")

        logger.info("Starting crew execution...")
        result = crew.kickoff()
        logger.info("Crew execution completed.")
        
        return result
