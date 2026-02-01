#!/bin/bash

# Models to benchmark
MODELS=(
    "openai/gpt-4o"
    "anthropic/claude-3.5-sonnet"
    "meta-llama/llama-3.3-70b-instruct"
    "mistralai/mistral-large-2411"
)

# Benchmark parameters
N=100
BASE_CMD="python3 benchmarks/ieee_repro/benchmark_tools.py"

echo "Starting Multi-Model Benchmark (N=$N)"
echo "-----------------------------------"

for model in "${MODELS[@]}"; do
    # Create safe directory name from model ID
    safe_name=$(echo $model | tr '/' '_')
    out_dir="benchmarks/ieee_repro/results/tools/$safe_name"
    
    echo "Running $model..."
    echo "Output: $out_dir"
    
    mkdir -p "$out_dir"
    
    $BASE_CMD --n $N --model "$model" --output "$out_dir" > "$out_dir/run.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $model finished."
        # Extract quick stats
        grep "C3 (Tool Input):" "$out_dir/run.log"
        grep "C6 (Logs):" "$out_dir/run.log"
    else
        echo "❌ $model failed. Check $out_dir/run.log"
    fi
    echo "-----------------------------------"
done

echo "🎉 All benchmarks completed."
