import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

class BaseModel:
    def __init__(self, tokenizer_name, model_name, lora_setting=None) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, 
                                                device_map="auto", 
                                                trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        if lora_setting is not None:
            lora_dropout = lora_setting['dropout']
            lora_alpha = lora_setting['alpha']
            lora_r = lora_setting['lora_r']

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                lora_dropout=lora_dropout,
                lora_alpha=lora_alpha,
                r=lora_r,
                bias="none",
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
            )

            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

            model.print_trainable_parameters()
        self.model = model
        self.peft_config = peft_config


    