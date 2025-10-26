import json
import numpy as np
logdir = f'eval/'
# logfile = <your_logfile>
logfile='/orcd/data/jhm/001/annesyab/LLM/AI_safety/llm-attacks/experiments/eval_scripts/eval/proprietary_transfer_vicuna_gcg_25_progressive_20251015-10:07:24.json'

with open(logfile, 'r') as f:
    log = json.load(f)
    
jb, em = [],[]
for model in log:
    stats = log[model]
    jb.append(stats['test_jb'])
    # em.append(stats['test_em'])
jb = np.array(jb)
jb = jb.mean(-1)
# em = np.array(em)
# em = em.mean(-1)
print(jb)
# print(em)