import argparse
from utils.trainer import DpoTraining



def dpo():
    parser = argparse.ArgumentParser(description='DPO finetune a model.')
    parser.add_argument('-d', '--dataset', default="/root/Self-Reward-RLHF/generated/preferences.jsonl", type=str, help='input sft dataset')
    parser.add_argument('-b', '--base_model', default="Dudep/phi2-sft", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-m', '--model', default="Dudep/phi2-sft", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-o', '--output', default="/root/Self-Reward-RLHF/generated-model", type=str, help='output trained model')
    args = parser.parse_args()


    lr=5e-5
    batch_size = 4
    lora_setting = {
    'dropout':0.05,
    'alpha':16,
    'lora_r':16
    }

    trainer = DpoTraining(
        tokenizer_name=args.base_model,
        model_name=args.model,
        data_path=args.dataset,
        output_path=args.output,
        lora_setting=lora_setting,
        lr=lr,
        batch_size=batch_size,
    )
    trainer()

if __name__ == "__main__":
    dpo()