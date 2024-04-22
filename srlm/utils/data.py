import os
import sys


def chat_format(example):
    if len(example["prompt"])>0:
        prompt="<|im_start|>system\nYou are an AI therapist. You will be given a task. You must generate an response for the user query.<|im_end|>\n<|im_start|>User\n"+example["prompt"]+"<|im_end|>\n<|im_start|>assistant\n"
    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>\n"
    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"
    return {"prompt":prompt,"chosen":chosen,"rejected":rejected}

def collate_fn(tokenizer, x):
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": x['prompt']},
        {"role": "assistant", "content": x['completion']},
    ], tokenize=False)
    return {"text": text}