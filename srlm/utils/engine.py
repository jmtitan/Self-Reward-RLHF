import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class GenEngine:

    def __init__(self, tokenizer_name, model_name):

        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16,
                                        )   
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, 
                                              device_map="auto", 
                                              trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.model.eval()

    def sample(self, input):
        pass
    
    def generate_prompts(self, input_path, output_path):
        pass

    def generate_responses(self, input_path, output_path):
        pass

    def generate_scores(self, input_path, output_path):
        pass

    def generate_preferences(self, input_path, output_path):
        pass
