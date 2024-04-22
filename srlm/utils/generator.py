
import re
from engine import GenEngine
from get_prompts import *
from get_responses import *


class Generator(GenEngine):
    
    device = "cuda"
    prompts_num = 1000

    
    srlm_prompt = """Review the user’s question and the corresponding response using the additive 5-point
    scoring system described below. 

    The user's question is between <question> and </question>
    The response of the AI Assistant is between <response> and </response>

    Points are accumulated based on the satisfaction of each
    criterion:
    - Add 1 point if the response is relevant and provides some information related to
    the user’s inquiry, even if it is incomplete or contains some irrelevant content.
    - Add another point if the response addresses a substantial portion of the user’s question,
    but does not completely resolve the query or provide a direct answer.
    - Award a third point if the response answers the basic elements of the user’s question in a
    useful way, regardless of whether it seems to have been written by an AI Assistant or if it
    has elements typically found in blogs or search results.
    - Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
    addressing the user’s question directly and comprehensively, and is well-organized and
    helpful, even if there is slight room for improvement in clarity, conciseness or focus.
    - Bestow a fifth point for a response that is impeccably tailored to the user’s question
    by an AI Assistant, without extraneous information, reflecting expert knowledge, and
    demonstrating a high-quality, engaging, and insightful answer.
    - If the response repeats itself or is not concise and to the point, score the response 0.

    <question>{prompt}</question>
    <response>{response}</response>

    After examining the user’s instruction and the response:
    - output the score of the evaluation using this exact format: "score: <total points>", where <total points> is between 0 and 5
    - Briefly justify your total score, up to 100 words.
    """

    def sample(self, inputs, generate_settings):

        if isinstance(inputs, str):
            """
            inputs = generate_prompt(examples)
            """
            print("<"*80)
            print(f"{inputs}")
            print(">"*80)
            model_inputs = self.tokenizer(inputs, return_tensors="pt").to("cuda")
        elif isinstance(inputs, list):
            """
            inputs = [
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": ""},
                    ]       
            """

            prompt_for_model = self.tokenizer.apply_chat_template(inputs, tokenize=False)

            model_inputs = self.tokenizer(prompt_for_model, return_tensors="pt").to("cuda")

        streamer = TextStreamer(self.tokenizer)

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            streamer=streamer,
            top_p=generate_settings['top_p'],
            temperature=generate_settings['temperature'],
            max_new_tokens=generate_settings['max_new_tokens']
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]

        return answer
    



    def generate_prompts(self, input_path, output_path):
        generate_settings = {
            "top_p": 0.9,
            "temperature": 0.6,
            "max_new_tokens": 256
        }

        ift_df = read_jsonl_file(input_path)
        uniq_prompts = set([])
        new_prompts = []
        while True:
            if len(uniq_prompts) >= self.prompts_num:
                break

            task_prompts = get_random_prompts(ift_df)
            inputs = generate_prompt(task_prompts)
            answer = self.sample(inputs, generate_settings)
            prompts = extract_prompts(answer)

            for prompt in prompts:
                if prompt not in uniq_prompts:
                    uniq_prompts.add(prompt)
                    prompt_id = str(uuid.uuid4())
                    new_prompts.append({"prompt_id": prompt_id, "prompt": prompt, "source": "generated"})

            new_prompts_df = pd.DataFrame(new_prompts)
            save_to_jsonl(new_prompts_df, output_path)

        return new_prompts_df.head()




    def generate_responses(self, input_path, output_path):
        generate_settings = {
            "top_p": 0.9,
            "temperature": 0.7,
            "max_new_tokens": 256
        }

        df_prompts = pd.read_json(path_or_buf=input_path, lines=True)
        # df_prompts = df_prompts.sample(100).reset_index(drop=True)
        # shuffle the dataframe
        df_prompts = df_prompts.sample(frac=1).reset_index(drop=True)

        completions = []
        for index, row in df_prompts.iterrows():
            print("============================================================================")
            print(f"Processing prompt {index + 1} of {len(df_prompts)}")

            prompt = row['prompt']
            prompt_id = row['prompt_id']
            buf = {"prompt_id": prompt_id, "prompt": prompt, "completion": {}}
            # sample 4 times as mentioned in the paper
            for idx in range(4):
                print("-----------------------------------------------------------------------")
                print(f"Processing prompt {index + 1}, completion {idx + 1}")
                inputs = [{"role": "user", "content": prompt},]
                answer = self.sample(inputs, generate_settings)
                completion = filter_completion(answer)

                print("\n\n")
                print(f"Extracted completion: {completion}")
                buf["completion"][idx] = completion

            completions.append(buf)

            df_completions = pd.DataFrame(completions)
            df_completions.to_json(output_path, orient='records', lines=True)
        
        return df_completions.head()




    def generate_scores(self, input_path, output_path):
        generate_settings = {
            "top_p": 1.0,
            "temperature": 1.0,
            "max_new_tokens": 100
        }

        df = pd.read_json(path_or_buf=input_path, lines=True)

        pattern = r"[Ss]core: ([0-5])"
        results = []
        for index, row in df.iterrows():
            prompt_id = row['prompt_id']
            prompt = row['prompt']
            completion = row['completion']

            print("-------------------------")

            llm_as_a_judge_prompt = self.srlm_prompt.format(prompt=prompt,response=completion)
            inputs = [{"role": "user", "content": llm_as_a_judge_prompt},]
            answer = self.sample(inputs, generate_settings)

            matches = re.findall(pattern, answer)
            generated_score = int(matches[0]) if matches else -1

            # print(f"Answer {answer}")
            print("Found Score: ", generated_score)

            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "completion": completion,
                "score": generated_score,
                "reasoning": answer
            })

            # save every time
            df_results = pd.DataFrame(results)
            df_results.to_json(output_path, orient='records', lines=True)


        return df_results.head()


    def generate_preferences(self, input_path, output_path):

        prompts = {}
        with open(input_path, "r") as f:
            for line in f:
                row = json.loads(line)
                prompt_id = row['prompt_id']

                if prompt_id not in prompts:
                    prompts[prompt_id] = []

                prompts[row['prompt_id']].append(row)

        pairs = []
        for prompt_id, prompts in prompts.items():
            # find the best score
            best_score = -1
            best_prompt = None
            for prompt in prompts:
                if prompt['score'] > best_score:
                    best_score = prompt['score']
                    best_prompt = prompt
            # find the worst score
            worst_score = 100
            worst_prompt = None
            for prompt in prompts:
                if prompt['score'] < worst_score:
                    worst_score = prompt['score']
                    worst_prompt = prompt

            if None == best_prompt or None == worst_prompt:
                continue

            if best_score == worst_score:
                continue

            pairs.append({
                "prompt_id": best_prompt['prompt_id'],
                "prompt": best_prompt['prompt'],
                "chosen": best_prompt['completion'],
                "rejected": worst_prompt['completion'],
                "score_chosen": best_prompt['score'],
                "score_rejected": worst_prompt['score']
            })



        with open(output_path, "w") as f:
            for line in pairs:
                f.write(json.dumps(line) + "\n")

        

    