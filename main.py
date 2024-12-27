# from opro_optimizer.settings import settings
from opro_optimizer.utils import call_hf_model
from transformers import AutoTokenizer, AutoModelForCausalLM

call_hf_model("hello world", tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct"),
      model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="auto"), max_decode_steps=20, temperature=0.8)