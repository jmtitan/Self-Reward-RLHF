
# code from https://github.com/Oxen-AI/Self-Rewarding-Language-Models.git

def generate_prompt(examples):
    prompt = """Come up with a series of tasks and questions.
    Only the task/question, no further text/explanation, no additional information.
    The task or question should be something a person would ask a chatbot.
    """
    for _, item in enumerate(examples):
        prompt += f"<task>{item}</task>\n"

    return prompt

def get_random_prompts(df, num_selections=8):
    all_selected_prompts = df.sample(n=num_selections)['prompt'].tolist()

    return all_selected_prompts

def extract_prompts(answer):
    # find all the prompts between <task> </task> brackets
    print("="*80)
    print("Extracting prompts...")
    print(answer)
    print("="*80)

    prompts = []
    while True:
        pattern = f"<task>"
        start = answer.find(pattern)
        if start == -1:
            break
        end = answer.find("</task>")
        if end == -1:
            break
        prompts.append(answer[start + len(pattern):end])
        answer = answer[end + len("</task>"):]

    print("Prompts extracted:")
    print(prompts)
    return prompts


def filter_completion(answer):
    # get pure anwer
    pattern = f"<|im_start|>assistant"
    parts = answer.split(pattern)
    if len(parts) < 1:
        raise ValueError("Wrong Answer format!")
    
    # del indicator
    completion = parts[1].replace("<|im_end|>", "")

    # find the last newline character and remove everything after it
    if "\n" in completion:
        last_newline = completion.rfind("\n")
        completion = completion[:last_newline]
        return completion.strip()
    else:
        return completion