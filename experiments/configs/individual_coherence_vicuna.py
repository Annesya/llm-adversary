import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    
    config.individual_coherence = True
    config.perplexity_weight=0.3
    
    return config