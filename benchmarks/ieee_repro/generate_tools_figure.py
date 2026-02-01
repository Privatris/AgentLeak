import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent / "results/tools"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "paper/figures"

# Model display names mapping
MODEL_MAP = {
    "openai_gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "anthropic_claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "mistralai_mistral-large-2411": "Mistral Large",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B"
}

# Order to display
ORDER = [
    "gpt-4o-mini", 
    "openai_gpt-4o", 
    "anthropic_claude-3.5-sonnet", 
    "mistralai_mistral-large-2411", 
    "meta-llama_llama-3.3-70b-instruct"
]

def load_data():
    models = []
    c3_scores = []
    c6_scores = []
    
    # Check what directories exist
    print(f"Scanning {RESULTS_DIR}...")
    
    for dir_name in ORDER:
        dir_path = RESULTS_DIR / dir_name
        claims_file = dir_path / "claims.json"
        
        display_name = MODEL_MAP.get(dir_name, dir_name)
        
        if claims_file.exists():
            try:
                data = json.loads(claims_file.read_text())
                claims = data.get("claims", {})
                
                c3 = claims.get("C3_tool_input", {}).get("rate", 0)
                c6 = claims.get("C6_logs", {}).get("rate", 0)
                
                models.append(display_name)
                c3_scores.append(c3)
                c6_scores.append(c6)
                print(f"Loaded {display_name}: C3={c3}%, C6={c6}%")
            except Exception as e:
                print(f"Error reading {claims_file}: {e}")
        else:
            print(f"Warning: No claims.json found for {dir_name}")
            
    return models, c3_scores, c6_scores

def generate_figure(models, c3_scores, c6_scores):
    if not models:
        print("No data to plot!")
        return

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors matching the theme
    rects1 = ax.bar(x - width/2, c3_scores, width, label='C3: Tool Input', color='#3498db', alpha=0.9)
    rects2 = ax.bar(x + width/2, c6_scores, width, label='C6: Logs (Thought Trace)', color='#e74c3c', alpha=0.9)

    # Styling
    ax.set_ylabel('Leakage Rate (%)')
    ax.set_title(f'Secondary Channel Leakage by Model (n=100)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110) # Little space for labels
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / 'tools_leakage.pdf'
    png_path = OUTPUT_DIR / 'tools_leakage.png'
    
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    print(f"Figures saved to:\n- {pdf_path}\n- {png_path}")

if __name__ == "__main__":
    models, c3, c6 = load_data()
    generate_figure(models, c3, c6)
