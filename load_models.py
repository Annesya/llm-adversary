from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lmsys/vicuna-13b-v1.5"  # <-- replace with your target model, e.g., "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
