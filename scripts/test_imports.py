import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agentleak.harness.adapters.langchain_adapter import LangChainAdapter
    print("LangChainAdapter imported successfully")
except ImportError as e:
    print(f"LangChainAdapter import failed: {e}")

try:
    from agentleak.harness.adapters.crewai_adapter import CrewAIAdapter
    print("CrewAIAdapter imported successfully")
except ImportError as e:
    print(f"CrewAIAdapter import failed: {e}")
