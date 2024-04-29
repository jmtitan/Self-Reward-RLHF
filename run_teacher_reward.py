import argparse
from srlm.generate.gen_utils.generator import Generator
from srlm.finetune import DPO, SFT


def run():
    parser = argparse.ArgumentParser(description='DPO finetune a model.')
    parser.add_argument('-d', '--dataset', default="data/M0/train/ift.jsonl", type=str, help='input sft dataset')
    parser.add_argument('-m', '--model', default="microsoft/phi-2", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-o', '--output', default="data/M0/generated/", type=str, help='output trained model')
    parser.add_argument('-u', '--user', required=True, type=str, help='huggingface user name')
    args = parser.parse_args()




    # SFT fine-tuning
    sft_model_name = SFT.train(args)

    # Data generation
    ift_dataset_file = args.dataset
    generated_prompts_file = args.output + "prompts.jsonl"
    generated_responses_file = args.output+ "responses.jsonl"
    generated_scores_file = args.output+ "gemini_scores.jsonl"
    generated_prefer_file = args.output+ "gemini_rew_preferences.jsonl"

    Gen = Generator(args.base_model, args.model)

    Gen.generate_prompts(ift_dataset_file, generated_prompts_file)
    Gen.generate_responses(generated_prompts_file, generated_responses_file)
    Gen.gemini_as_judge(generated_responses_file, generated_scores_file)    
    Gen.generate_preferences(generated_scores_file, generated_prefer_file)

    # DPO
    args.model = sft_model_name
    dpo_model_name = DPO.train(args)
    print(f'Model uploaded in Huggingface\nSFT model name:{sft_model_name}\nDPO model name {dpo_model_name}' )

