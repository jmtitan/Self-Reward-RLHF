import argparse
from gen_utils.generator import Generator



def run():
    parser = argparse.ArgumentParser(description='DPO finetune a model.')
    parser.add_argument('-d', '--dataset', default="data/M0/train/ift.jsonl", type=str, help='input sft dataset')
    parser.add_argument('-b', '--base_model', default="Dudep/phi2-sft", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-m', '--model', default="Dudep/phi2-sft", type=str, help='the base model we want to fine-tune')
    parser.add_argument('-o', '--output', default="data/M0/generated/", type=str, help='output trained model')
    args = parser.parse_args()

    ift_dataset_file = args.dataset
    generated_prompts_file = args.output + "prompts.jsonl"
    generated_responses_file = args.output+ "responses.jsonl"
    generated_scores_file = args.output+ "scores.jsonl"
    generated_prefer_file = args.output+ "preferences.jsonl"

    Gen = Generator(args.base_model, args.model)

    Gen.generate_prompts(ift_dataset_file, generated_prompts_file)
    Gen.generate_responses(generated_prompts_file, generated_responses_file)
    Gen.generate_scores(generated_responses_file, generated_scores_file)    
    Gen.generate_preferences(generated_scores_file, generated_prefer_file)

if __name__ == "__main__":
    run()
