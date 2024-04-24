import os
from utils.model import BaseModel
from utils.data import collate_fn, chat_format
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DPOTrainer


class Trainer(BaseModel):

    def __init__(self, tokenizer_name, model_name, data_path, output_path, 
                 lora_setting=None) -> None:
        super().__init__(tokenizer_name, model_name, lora_setting)

        # load the training dataset
        dataset = load_dataset("json", data_files={'train': data_path})
        dataset = dataset['train'].shuffle(seed=42)
        # data mapping
        self.dataset = dataset
        
        self.output_path = output_path


    def __call__(self):
        pass
        


class SftTraining(Trainer):

    def __init__(self, tokenizer_name, model_name, data_path, output_path, lora_setting=None, lr=0.0002, batch_size=4) -> None:
        super().__init__(tokenizer_name, model_name, data_path, output_path, lora_setting)
        
        self.dataset.map(lambda x: collate_fn(self.tokenizer, x))
        self.model.config.pretraining_tp = 1
        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        

    def __call__(self):
        training_args = TrainingArguments(
            output_dir=self.output_path,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            gradient_accumulation_steps=4,
            warmup_steps=30,
            logging_steps=1,
            num_train_epochs=1,
            save_steps=50,
            remove_unused_columns=True
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            max_seq_length=1024,
            tokenizer=self.tokenizer,
            args=training_args,
            dataset_text_field="text"
        )

        trainer.train()

        output_dir = os.path.join(self.output_path, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)

class DpoTraining(Trainer):

    def __init__(self, tokenizer_name, model_name, data_path, output_path, lora_setting=None, lr=0.0002, batch_size=4) -> None:
        super().__init__(tokenizer_name, model_name, data_path, output_path, lora_setting)
        
        self.dataset.map(chat_format)
        self.model.config.use_cache = False

        # hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        

    def __call__(self, ref_model=None):
        training_args = TrainingArguments(
            output_dir=self.output_path,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            warmup_steps=50,
            logging_steps=1,
            num_train_epochs=1,
            save_steps=50,
            lr_scheduler_type="cosine",
            optim="paged_adamw_32bit",
            remove_unused_columns=True,
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            beta=0.1,
            max_prompt_length=1024,
            max_length=1536,
        )

        trainer.train()

        output_dir = os.path.join(self.output_path, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)