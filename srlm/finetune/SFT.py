
import argparse
from ft_utils.trainer import SftTraining



def train(args):


    lr=2e-4
    batch_size = 4
    lora_setting = {
    'dropout':0.1,
    'alpha':16,
    'lora_r':64
    }

    trainer = SftTraining(
        model_name=args.model,
        data_path=args.dataset,
        output_path=args.output,
        user_name=args.user,
        lora_setting=lora_setting,
        lr=lr,
        batch_size=batch_size,

    )
    trainer()
    model_name = trainer.push_to_hub()
    return model_name
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFT train a model.')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='input sft dataset')
    parser.add_argument('-m', '--model', default="microsoft/phi-2", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-o', '--output', required=True, type=str, help='output trained model')
    parser.add_argument('-u', '--user', required=True, type=str, help='huggingface user name')
    args = parser.parse_args()

    train(args)
