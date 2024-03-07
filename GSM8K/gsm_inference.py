from openai import OpenAI
import json
import numpy as np
import random
import time
from tqdm import tqdm
import argparse
import pandas as pd
from tqdm import tqdm
from mlx_lm import load, generate

client = None

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument(
        "--API_KEY",
        type=str,
        help="your OpenAI API key to use gpt-3.5-turbo"
    )
    parser.add_argument("--round", default=2, type=int)
    parser.add_argument(
        "--cot",
        default=False,
        action='store_true',
        help="If this is True, you can use Chain-of-Thought during inference."
    )
    parser.add_argument(
        "--output_dir",
        default="Math",
        type=str,
        help="Directory to save the result file"
    )

    return parser.parse_args()


def generate_response(input, model, tokenizer, temp=0, max_tokens=256):
    """
    Generate a response using the Mistral model with a given inner setting, prompt, and question string.
    """

    response = generate(model, tokenizer, input, temp=temp, max_tokens=max_tokens)  # MAX_TOKENS is a critical parameters to avoid annecessary additional text
    return response.strip()

def load_json(prompt_path, endpoint_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(endpoint_path, "r") as endpoint_file:
        endpoint_dict = json.load(endpoint_file)

    return prompt_dict, endpoint_dict

def construct_message(agent_context, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    prefix_string = prefix_string + agent_context + "\n\n Write a summary of the different opinions from each of the individual agent."

    message = [{"role": "user", "content": prefix_string}]

    try:
        completion = client.chat.completions.create(model="gpt-3.5-turbo-0613",
        messages=message,
        max_tokens=256,
        n=1)
        summary = completion.choices[0].message.content
        print(summary)
    except:
        print("retrying ChatGPT due to an error......")
        time.sleep(5)
        return construct_message(agent_context, instruction, idx)

    prefix_string = f"Here is a summary of responses from other agents: {summary}"
    prefix_string = prefix_string + "\n\n Use this summarization carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response." + instruction
    return prefix_string

def summarize_message(agent_contexts, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    completion = construct_message(prefix_string, instruction, idx)

    return completion

def generate_gsm(agents, question):
    agent_contexts = [[{"model": agent, "content": f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."}] for agent in agents]
    return agent_contexts

def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_model_and_tokenizer(model_name):

    if model_name == 'mixtral':
        model, tokenizer = load("mlx-community/Mixtral-8x7B-Instruct-v0.1-hf-4bit-mlx") 
    elif model_name == 'gemma2B':
        model, tokenizer = load("mlx-community/quantized-gemma-2b-it")
    elif model_name == 'gemma7B':
        model, tokenizer = load("mlx-community/quantized-gemma-7b-it")
    elif model_name == 'hermes':
        model, tokenizer = load("mlx-community/Nous-Hermes-2-Mixtral-8x7B-DPO-4bit")
    elif model_name == "llama":
        model, tokenizer = load("mlx-community/Llama-2-7b-chat-4-bit")
    elif model_name == "tinyllama":
        model, tokenizer = load("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")
    elif model_name == "llama-pro":
        model, tokenizer = load("mlx-community/LLaMA-Pro-8B-Instruct-mlx")       
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return (model, tokenizer)

if __name__ == "__main__":
    args = args_parse()
    model_list = [args.model_1, args.model_2, args.model_3]
    client = OpenAI(api_key=args.API_KEY)

    model_dict = {}
    for i,mdl in enumerate(model_list):
        model_dict[model_list[i]] = get_model_and_tokenizer(model_list[i])

    prompt_dict, endpoint_dict = load_json("src/prompt_template.json", "src/inference_endpoint.json")

    def generate_answer(model_name, formatted_prompt):
        # API_URL = endpoint_dict[model]["API_URL"]
        # headers = endpoint_dict[model]["headers"]
        # payload = {
        #     "inputs": formatted_prompt,
        #     "parameters": {
        #         "max_new_tokens": 256
        #     }
        # }

        # inputs = "" 
        # if args.model == 'mixtral':
        #     inputs = f"<s>[INST] {formatted_prompt}\n{questions_string} [/INST]"
        # elif args.model == 'gemma2B' or args.model == 'gemma7B':
        #     inputs = f"<start_of_turn>user\n{formatted_prompt}\n{questions_string}<end_of_turn>\n<start_of_turn>model"
        # elif args.model == 'hermes':
        #     inputs = f"<|im_start|>system\n{formatted_prompt}<|im_start|>user\n{questions_string}<|im_end|>\n<|im_start|>system"
        # else:
        #     raise ValueError(f"Model {args.model} not recognized.")


        model, tokenizer = model_dict[model_name]

        try:
            resp = generate_response(formatted_prompt, model, tokenizer)
            # resp = requests.post(API_URL, json=payload, headers=headers)
            response = resp
        except:
            print("retrying due to an error......")
            time.sleep(5)
            return generate_answer(model, formatted_prompt)
        
        return {"model": model_name, "content": response}
    
    def prompt_formatting(model, instruction, cot):
        if model == "alpaca" or model == "orca":
            prompt = prompt_dict[model]["prompt_no_input"]
        else:
            prompt = prompt_dict[model]["prompt"]
        
        if cot:
            instruction += "Let's think step by step."

        return {"model": model, "content": prompt.format(instruction=instruction)}

    agents = len(model_list)
    rounds = args.round
    random.seed(0)

    evaluation = 100

    generated_description = []

    questions = read_jsonl("data/GSM8K/gsm8k_test.jsonl")
    random.shuffle(questions)

    for idx in tqdm(range(evaluation)):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        agent_contexts = generate_gsm(model_list, question)

        print(f"# Question No.{idx+1} starts...")

        message = []

        # Debate
        for debate in range(rounds+1):
            # Refer to the summarized previous response
            if debate != 0:
                message.append(summarize_message(agent_contexts, question, 2 * debate - 1))
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(prompt_formatting(agent_contexts[i][-1]["model"], message, args.cot))

            for agent_context in agent_contexts:
                # Generate new response based on summarized response
                completion = generate_answer(agent_context[-1]["model"], agent_context[-1]["content"])
                agent_context.append(completion)

        print(f"# Question No.{idx+1} debate is ended.")

        models_response = {
            f"{args.model_1}": [agent_contexts[0][1]["content"], agent_contexts[0][3]["content"], agent_contexts[0][-1]["content"]],
            f"{args.model_2}": [agent_contexts[1][1]["content"], agent_contexts[1][3]["content"], agent_contexts[1][-1]["content"]],
            f"{args.model_3}": [agent_contexts[2][1]["content"], agent_contexts[2][3]["content"], agent_contexts[2][-1]["content"]]
        }
        response_summarization = [
            message[0], message[1]
        ]
        generated_description.append({"question_id": idx, "question": question, "agent_response": models_response, "summarization": response_summarization, "answer": answer})

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    print(f"The result file 'gsm_result{file_name}' is saving...")
    with open(args.output_dir + f"/gsm_result{file_name}", "x") as f:
        json.dump(generated_description, f, indent=4)

    print("All done!!")