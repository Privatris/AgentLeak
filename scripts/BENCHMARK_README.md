# AgentLeak Benchmark Suite

## 🎯 Overview

The AgentLeak Benchmark Suite is a professional testing framework for evaluating privacy leakage in LLM-powered agents. It supports:

- **23 LLM Models** across 7 providers (OpenAI, Anthropic, Google, Meta, Alibaba, Mistral, DeepSeek)
- **5 Multi-Agent Frameworks** (LangChain, CrewAI, AutoGPT, MetaGPT, AgentGPT)
- **7 Leakage Channels** (C1-C7: Final Output, Inter-Agent, Tool I/O, Memory, Logs, Artifacts)
- **3 Attack Levels** (A0 Benign, A1 Indirect Injection, A2 Adversarial)
- **4 Domain Verticals** (Healthcare, Finance, Legal, Corporate)

## 📊 Quick Start

### Prerequisites

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"

# Install dependencies
pip install requests
```

### Running Benchmarks

```bash
# Quick validation test (~10 scenarios, ~$0.10)
./scripts/run_full_benchmark.sh quick

# Standard benchmark (~100 scenarios, ~$5)
./scripts/run_full_benchmark.sh standard

# Full paper benchmark (~1000 scenarios, ~$560)
./scripts/run_full_benchmark.sh paper
```

### Dry Run (Preview without executing)

```bash
./scripts/run_full_benchmark.sh paper --dry-run
```

## 📦 Benchmark Profiles

| Profile | Models | Scenarios | Est. Cost | Description |
|---------|--------|-----------|-----------|-------------|
| `quick` | 2 | 10 | ~$0.10 | Fast validation |
| `standard` | 4 | 100 | ~$5 | Balanced coverage |
| `comprehensive` | 23 | 500 | ~$100 | Full model coverage |
| `flagship` | 9 | 200 | ~$50 | Top-tier models only |
| `paper` | 10 | 1000 | ~$560 | Publication-ready |

## 🤖 Available Models

### Flagship Tier (Best Performance)
- **OpenAI**: GPT-4o, GPT-4 Turbo, o1, o1-mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude Sonnet 4
- **Google**: Gemini 2.5 Pro, Gemini 2.0 Flash

### Mid Tier (Good Balance)
- **OpenAI**: GPT-4o Mini
- **Anthropic**: Claude 3.5 Haiku
- **Meta**: Llama 3.3 70B, Llama 3.1 70B
- **Alibaba**: Qwen 2.5 72B
- **Mistral**: Mistral Large
- **DeepSeek**: DeepSeek Chat, DeepSeek R1

### Budget Tier (Cost Efficient)
- **Alibaba**: Qwen 2.5 7B, Qwen 2.5 Coder 32B
- **Meta**: Llama 3.1 8B
- **Google**: Gemini Flash 1.5 8B
- **Mistral**: Mistral 7B, Mixtral 8x7B

## 📈 Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **TSR** | Task Success Rate | ↑ Higher is better |
| **ELR** | Exact Leakage Rate | ↓ Lower is better |
| **WLS** | Weighted Leakage Score | ↓ Lower is better |
| **CLR** | Channel Leakage Rate (per-channel) | ↓ Lower is better |
| **ASR** | Attack Success Rate | ↓ Lower is better |

## 📁 Output Structure

```
benchmark_results/
├── benchmark_20251226_123456.json    # Full results
├── traces_20251226_123456.jsonl      # Individual traces
├── summary_20251226_123456.txt       # Summary table
├── latex/
│   ├── report_*_model_comparison.tex
│   ├── report_*_channel_breakdown.tex
│   ├── report_*_attack_comparison.tex
│   └── report_*_pareto_analysis.tex
├── csv/
│   ├── report_*_models.csv
│   ├── report_*_channels.csv
│   └── report_*_attacks.csv
└── report_*.md                       # Markdown report
```

## 🔧 Custom Configuration

```bash
# Test specific models
python scripts/full_benchmark.py \
    --models gpt-4o claude-3.5-sonnet llama-3.3-70b \
    --frameworks langchain_single crewai \
    --n-scenarios 50 \
    --attacks A0 A1 A2

# Generate reports from results
python scripts/benchmark_reporter.py benchmark_results/benchmark_*.json

# Compare multiple runs
python scripts/benchmark_reporter.py --compare run1.json run2.json
```

## 📊 Sample Output

```
================================================================================
AGENTLEAK BENCHMARK RESULTS
Timestamp: 20251226_143052
================================================================================

MODEL COMPARISON
--------------------------------------------------------------------------------
Model                     TSR      ELR      WLS    Tokens
--------------------------------------------------------------------------------
claude-3.5-sonnet        92.4%   18.3%     0.42     125,432
gpt-4o                   89.1%   22.7%     0.58     142,891
gemini-2.5-pro           87.5%   24.1%     0.61     118,234
llama-3.3-70b            85.2%   28.9%     0.73     156,782
--------------------------------------------------------------------------------

OVERALL METRICS
----------------------------------------
Total Scenarios:        1000
Overall TSR:           88.6%
Overall ELR:           23.5%
Average WLS:            0.59
================================================================================
```

## 📝 Generating Paper Tables

The benchmark automatically generates LaTeX tables ready for academic papers:

```latex
\begin{table}[htbp]
\centering
\caption{Model Privacy Leakage Comparison across AgentLeak Benchmark}
\label{tab:model-comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{TSR} $\uparrow$ & \textbf{ELR} $\downarrow$ & ...
\midrule
claude-3.5-sonnet & 92.4\% & 18.3\% & 0.42 & ...
...
\bottomrule
\end{tabular}
\end{table}
```

## 🔬 Channel Descriptions

| Channel | Name | Risk Level | Description |
|---------|------|------------|-------------|
| C1 | Final Output | HIGH | User-visible response |
| C2 | Inter-Agent | HIGH | Messages between agents |
| C3 | Tool Input | MEDIUM | Arguments to tool calls |
| C4 | Tool Output | MEDIUM | Results from tools |
| C5 | Memory | MEDIUM | Persistent stores |
| C6 | Logs | LOW | Debug traces |
| C7 | Artifacts | HIGH | Generated files |

## ⚔️ Attack Levels

| Level | Name | Description |
|-------|------|-------------|
| A0 | Benign | No attack - legitimate user request |
| A1 | Indirect | Hidden instructions in tool outputs |
| A2 | Adversarial | Multi-turn jailbreak and social engineering |

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{agentleak2025,
  title={AgentLeak: A Benchmark for Privacy Leakage in LLM-Powered Multi-Agent Systems},
  author={...},
  booktitle={Proceedings of...},
  year={2025}
}
```

## 📄 License

MIT License - See LICENSE file for details.
