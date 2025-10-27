# import json
# import numpy as np
# logdir = f'eval/'

# # use the json files eval_scripts/eval_baseline, eval_baseline_sufix, eval_gcg  to plot the test jailbreak rates for each model on a single plot

# logfile='/orcd/data/jhm/001/annesyab/LLM/AI_safety/llm-attacks/experiments/eval_scripts/eval/proprietary_transfer_vicuna_gcg_25_progressive_20251015-10:07:24.json'

# with open(logfile, 'r') as f:
#     log = json.load(f)
    
# jb, em = [],[]
# for model in log:
#     stats = log[model]
#     jb.append(stats['test_jb'])
#     # em.append(stats['test_em'])
# jb = np.array(jb)
# jb = jb.mean(-1)
# # em = np.array(em)
# # em = em.mean(-1)
# print(jb)
# # print(em)

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# Base directory
base_dir = '/orcd/data/jhm/001/annesyab/LLM/AI_safety/llm-attacks/experiments/eval_scripts'

# Three evaluation directories
eval_dirs = {
    'Baseline': 'eval_baseline',
    'Baseline + Suffix': 'eval_baseline_suffix',
    'GCG': 'eval_gcg'
}

# Dictionary to store results: {model_name: {method: jb_rate}}
results = {}

# Read data from all three directories
for method_name, dir_name in eval_dirs.items():
    dir_path = os.path.join(base_dir, dir_name)
    json_files = glob(os.path.join(dir_path, '*.json'))
    
    for json_file in json_files:
        # Extract model name from filename
        filename = os.path.basename(json_file)
        model_name = filename.split('_transfer')[0]
        
        # Read JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract test jailbreak rate
        for model_key, stats in data.items():
            if 'final_test_jb_rate' in stats:
                jb_rate = stats['final_test_jb_rate'] * 100  # Convert to percentage
            elif 'test_jb' in stats:
                # Calculate average if final rate not available
                test_jb = np.array(stats['test_jb'])
                jb_rate = np.mean(test_jb) * 100
            else:
                continue
            
            # Store result
            if model_name not in results:
                results[model_name] = {}
            results[model_name][method_name] = jb_rate

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for plotting
models = sorted(results.keys())
methods = list(eval_dirs.keys())
x = np.arange(len(models))
width = 0.25  # Width of bars

# Plot bars for each method
for i, method in enumerate(methods):
    jb_rates = [results[model].get(method, 0) for model in models]
    ax.bar(x + i * width, jb_rates, width, label=method)

# Customize plot
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Jailbreak Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Test Jailbreak Rates Across Models and Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('jailbreak_rates_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nTest Jailbreak Rates Summary (%):\n")
print(f"{'Model':<25} {'Baseline':<15} {'Baseline + Suffix':<20} {'GCG':<10}")
print("-" * 75)
for model in models:
    baseline = results[model].get('Baseline', 0)
    baseline_suffix = results[model].get('Baseline + Suffix', 0)
    gcg = results[model].get('GCG', 0)
    print(f"{model:<25} {baseline:<15.2f} {baseline_suffix:<20.2f} {gcg:<10.2f}")
    
# 