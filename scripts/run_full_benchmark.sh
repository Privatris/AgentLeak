#!/bin/bash
# =============================================================================
# AgentLeak Full Benchmark Runner
# =============================================================================
#
# This script runs the complete AgentLeak benchmark suite with various profiles.
#
# Usage:
#   ./run_full_benchmark.sh                     # Show help
#   ./run_full_benchmark.sh quick               # Quick validation test
#   ./run_full_benchmark.sh standard            # Standard benchmark
#   ./run_full_benchmark.sh paper               # Full paper benchmark
#   ./run_full_benchmark.sh custom <args>       # Custom configuration
#
# Requirements:
#   - Python 3.9+
#   - OPENROUTER_API_KEY environment variable set
#   - Dependencies installed (pip install -r requirements.txt)
#
# Author: AgentLeak Research Team
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${BLUE}Loading environment from .env...${NC}"
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# =============================================================================
# FUNCTIONS
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗                ║"
    echo "║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝                ║"
    echo "║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║                   ║"
    echo "║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║                   ║"
    echo "║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║                   ║"
    echo "║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                   ║"
    echo "║                                                                ║"
    echo "║     ██╗     ███████╗ █████╗ ██╗  ██╗                           ║"
    echo "║    ██║     ██╔════╝██╔══██╗██║ ██╔╝                            ║"
    echo "║    ██║     █████╗  ███████║█████╔╝                             ║"
    echo "║    ██║     ██╔══╝  ██╔══██║██╔═██╗                             ║"
    echo "║    ███████╗███████╗██║  ██║██║  ██╗                            ║"
    echo "║    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                            ║"
    echo "║                                                                ║"
    echo "║              Professional Benchmark Suite v1.0                 ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    echo -e "${GREEN}Usage:${NC} $0 <command> [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  quick          Quick validation test (~10 scenarios, ~\$0.10)"
    echo "  standard       Standard benchmark (~100 scenarios, ~\$5)"
    echo "  comprehensive  Full benchmark (~500 scenarios, ~\$100)"
    echo "  flagship       Flagship models only (~200 scenarios, ~\$50)"
    echo "  paper          Paper-ready benchmark (~1000 scenarios, ~\$200)"
    echo "  custom         Custom configuration (pass additional args)"
    echo "  list-models    List all available models"
    echo "  list-profiles  List all benchmark profiles"
    echo "  estimate       Estimate costs for a profile"
    echo "  report         Generate reports from latest results"
    echo "  compare        Compare multiple benchmark runs"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --dry-run      Show what would be run without executing"
    echo "  --api-key KEY  Set OpenRouter API key"
    echo "  --output-dir   Set output directory"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 quick                          # Quick test"
    echo "  $0 standard --dry-run             # Preview standard benchmark"
    echo "  $0 custom --models gpt-4o claude-3.5-sonnet --n-scenarios 50"
    echo "  $0 report                         # Generate reports"
    echo ""
    echo -e "${YELLOW}Environment:${NC}"
    echo "  OPENROUTER_API_KEY   Your OpenRouter API key (required)"
    echo ""
}

check_requirements() {
    echo -e "${BLUE}Checking requirements...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 not found${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "  Python version: ${GREEN}$PYTHON_VERSION${NC}"
    
    # Check API key
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo -e "${YELLOW}Warning: OPENROUTER_API_KEY not set${NC}"
        echo "  Set it with: export OPENROUTER_API_KEY='your-key'"
    else
        echo -e "  API key: ${GREEN}Set (${#OPENROUTER_API_KEY} chars)${NC}"
    fi
    
    # Check project structure
    if [ ! -f "$PROJECT_ROOT/scripts/full_benchmark.py" ]; then
        echo -e "${RED}Error: full_benchmark.py not found${NC}"
        exit 1
    fi
    echo -e "  Project root: ${GREEN}$PROJECT_ROOT${NC}"
    
    echo ""
}

run_benchmark() {
    local profile=$1
    shift
    local extra_args="$@"
    
    echo -e "${BLUE}Running benchmark profile: ${GREEN}$profile${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    python3 scripts/full_benchmark.py --profile "$profile" $extra_args
    
    echo ""
    echo -e "${GREEN}Benchmark completed!${NC}"
}

run_custom() {
    echo -e "${BLUE}Running custom benchmark...${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    python3 scripts/full_benchmark.py "$@"
    
    echo ""
    echo -e "${GREEN}Benchmark completed!${NC}"
}

generate_reports() {
    echo -e "${BLUE}Generating reports...${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    python3 scripts/benchmark_reporter.py "$@"
    
    echo ""
    echo -e "${GREEN}Reports generated!${NC}"
}

estimate_cost() {
    local profile=$1
    
    echo -e "${BLUE}Estimating costs for profile: ${GREEN}$profile${NC}"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    python3 -c "
from scripts.benchmark_config import BENCHMARK_PROFILES, estimate_benchmark_cost

profile = BENCHMARK_PROFILES.get('$profile')
if not profile:
    print('Unknown profile: $profile')
    exit(1)

estimate = estimate_benchmark_cost(profile)

print(f'Profile: {profile.name}')
print(f'Description: {profile.description}')
print()
print(f'Configuration:')
print(f'  Models: {len(profile.models)}')
print(f'  Frameworks: {len(profile.frameworks)}')
print(f'  Scenarios: {profile.n_scenarios}')
print(f'  Attack Levels: {len(profile.attack_levels)}')
print()
print(f'Cost Estimate:')
print(f'  Total API Calls: {estimate[\"total_api_calls\"]:,}')
print(f'  Estimated Cost: \${estimate[\"total_estimated_cost\"]:.2f}')
print(f'  Budget Limit: \${profile.max_cost_usd:.2f}')
print(f'  Within Budget: {\"Yes\" if estimate[\"within_budget\"] else \"No\"}')"
}

# =============================================================================
# MAIN
# =============================================================================

print_banner

if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    quick|standard|comprehensive|flagship|paper)
        check_requirements
        run_benchmark "$COMMAND" "$@"
        ;;
    custom)
        check_requirements
        run_custom "$@"
        ;;
    list-models)
        cd "$PROJECT_ROOT"
        python3 scripts/full_benchmark.py --list-models
        ;;
    list-profiles)
        cd "$PROJECT_ROOT"
        python3 scripts/full_benchmark.py --list-profiles
        ;;
    estimate)
        PROFILE=${1:-standard}
        estimate_cost "$PROFILE"
        ;;
    report)
        generate_reports "$@"
        ;;
    compare)
        cd "$PROJECT_ROOT"
        python3 scripts/benchmark_reporter.py --compare "$@"
        ;;
    help|-h|--help)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}Done!${NC}"
