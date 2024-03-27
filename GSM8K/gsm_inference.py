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
args = None

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument(
        "--OPENAI_API_KEY",
        type=str,
        help="your OpenAI API key to use gpt-3.5-turbo"
    )
    parser.add_argument(
        "--GOOGLE_API_KEY",
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
        default="GSM8K",
        type=str,
        help="Directory to save the result file"
    )
    # parser.add_argument("--env", type=str) #macOS, cq

    return parser.parse_args()


def generate_response(input, model_name, model, tokenizer, temp=0, max_tokens=3000):
    """
    Generate a response using the Mistral model with a given inner setting, prompt, and question string.
    """

    if model_name == 'gemini-pro':
        return generate_response_gemini(input, model)
    elif model_name == "palm2":
        return generate_response_palm(input, model)
    else:
        response = generate(model, tokenizer, input, temp=temp, max_tokens=max_tokens)  # MAX_TOKENS is a critical parameters to avoid annecessary additional text
        return response.strip()

def generate_response_gemini(inputs, model, retry_count=0):
    """
    Generate a response using the Gemini model API,  with a given prompt, and question string.
    """
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    import google.generativeai as genai
    from google.api_core import retry
    genai.configure(api_key=args.GOOGLE_API_KEY)

    try:
        response = model.generate_content(
            inputs,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                # stop_sequences=['space'],
                # max_output_tokens=3000,
                temperature=0),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
        )
    except Exception as e:
        print(f"retrying Gemini due to an error: {e}")
        time.sleep(5)
        retry_count+=1
        if retry_count>10:
            return ""
        return generate_response_gemini(inputs, model, retry_count)

    return response.text.strip().replace('\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\\n\\',"").replace('\\\\\n\\\\\n\\\\\n',"")


def generate_response_palm(inputs, model, retry_count=0):
    """
    Generate a response using the Palm API, with a given prompt, and question string.
    """
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    import google.generativeai as genai
    genai.configure(api_key=args.GOOGLE_API_KEY)

    if not retry_count:
        retry_count = 0

    try:
        response = genai.generate_text(
            model=model,
            prompt=inputs,
            temperature=0,
            # max_output_tokens=256,
            safety_settings=[
                {
                    "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_VIOLENCE,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
            ]
        )
    except Exception as e:
        print(f"retrying Palm 2 due to an error: {e}")
        time.sleep(5)
        retry_count+=1
        if retry_count>10:
            return ""
            # raise Exception("Maximum retries exceeded for Palm 2") from e
        return generate_response_palm(inputs, model, retry_count)

    if not response.result:
        return ""
    else:
        return response.result.strip()

def load_json(prompt_path, endpoint_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(endpoint_path, "r") as endpoint_file:
        endpoint_dict = json.load(endpoint_file)

    return prompt_dict, endpoint_dict

def construct_message_gpt(question, agent_context, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    prefix_string = prefix_string + agent_context + " Write a summary of the different opinions from each of the individual agent."

    message = [{"role": "user", "content": prefix_string}]

    try:
        completion = client.chat.completions.create(model="gpt-3.5-turbo-0613",
        messages=message,
        max_tokens=256,
        n=1)
        summary = completion.choices[0].message.content
    except:
        print("retrying ChatGPT due to an error......")
        time.sleep(5)
        return construct_message_gpt(agent_context, instruction, idx)

    prefix_string = f"Here is a summary of responses from other agents: {summary} \n"
    prefix_string = prefix_string + "Use this summarization carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response." + instruction
    return prefix_string

def construct_message_gemini(agent_context, instruction, cot, idx):
    """
    Get summary from Gemini

    """
    inputs = f"<start_of_turn>user{agent_context}<end_of_turn><start_of_turn>model"
    model, _ = get_gemini_model("gemini-pro")
    summary = generate_response_gemini(inputs, model)

    prefix_string = f"{instruction} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. \n"
    prefix_string = prefix_string+ f"This question was previously asked to other AI agents and here is a summary of responses from other agents for inspiration:\n {summary}\n"
    prefix_string = prefix_string + " Use this summarization carefully as additional advice in solving the math problem, can you now provide an updated answer? Make sure to state your answer at the end of the response."
    if cot:
        prefix_string = prefix_string + "Let's think step by step." 
    return prefix_string

def summarize_message(agent_contexts, model_name, instruction, cot, idx):
    prefix_string = "You are a helpful AI Assistant that is an expert in summarization. Here are a list of opinions from different agents on the answer to a specific math question: \n"

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = " One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n Write a summary of the different opinions from each of the individual agents."
    if model_name == "gemini":
        summary = construct_message_gemini(prefix_string, instruction, cot, idx)
    else:
        summary = construct_message_gpt(prefix_string, instruction, idx)

    return summary

def generate_gsm(agents, question, is_cot):
    if is_cot:
        agent_contexts = [[{"model": agent, "content": f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. Let's think step by step."}] for agent in agents]
    else:
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
    elif model_name == "qwen":
        model, tokenizer = load("mlx-community/Qwen1.5-7B-Chat-4bit")
    elif model_name in ['gemini-pro', "palm2"]:
        model, tokenizer = get_gemini_model(model_name)      
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return (model, tokenizer)

def get_gemini_model(model_name):
    import google.generativeai as genai
    genai.configure(api_key=args.GOOGLE_API_KEY)
    tokenizer = None
    if model_name == 'gemini-pro':
        model = genai.GenerativeModel('gemini-pro')
    elif model_name == 'palm2':
        model = 'models/text-bison-001'
    return model, tokenizer

# def get_model_and_tokenize_for_calcul_q(model_name):
#     # TODO
#     model = None
#     tokenizer = None
#     # Qwen 14B + LLama 13B + Mixtral
#     # Qwen 72B + LLama 70B + 
#     return (model, tokenizer)

if __name__ == "__main__":
    args = args_parse()
    model_list = [args.model_1, args.model_2, args.model_3]
    # client = OpenAI(api_key=args.API_KEY)
    # import google.generativeai as genai
    # genai.configure(api_key=args.GOOGLE_API_KEY)

    model_dict = {}
    for i,mdl in enumerate(model_list):
        model_dict[model_list[i]] = get_model_and_tokenizer(model_list[i])

    prompt_dict, endpoint_dict = load_json("src/prompt_template.json", "src/inference_endpoint.json")

    def generate_answer(model_name, formatted_prompt, retries = 0):
        model, tokenizer = model_dict[model_name]
        max_retries = 10

        try:
            resp = generate_response(formatted_prompt,model_name, model, tokenizer)
            # resp = requests.post(API_URL, json=payload, headers=headers)
            response = resp
        except Exception as e:
            print(f"retrying due to an error: {e}")
            retries+=1
            if retries > max_retries:
                return {"model": model_name, "content": ""}
            time.sleep(5)
            return generate_answer(model_name, formatted_prompt, retries)
        
        return {"model": model_name, "content": response}
    
    def prompt_formatting(model, instruction, cot):
        if model == "alpaca" or model == "orca":
            prompt = prompt_dict[model]["prompt_no_input"]
        else:
            prompt = prompt_dict[model]["prompt"]
        
        # if cot:
        #     instruction += "Let's think step by step."

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

        agent_contexts = generate_gsm(model_list, question, args.cot)

        print(f"# Question No.{idx+1} starts...")

        message = []

        # Debate
        for debate in range(rounds+1):
            # Refer to the summarized previous response
            if debate != 0:
                # message.append(f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.")
                curr_message = summarize_message(agent_contexts, "gemini", question, args.cot, 2 * debate - 1)
                message.append(curr_message)
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(prompt_formatting(agent_contexts[i][-1]["model"], message, args.cot))

            for agent_context in agent_contexts:
                # Generate new response based on summarized response
                completion = generate_answer(agent_context[-1]["model"], agent_context[-1]["content"])
                agent_context.append(completion)

        print(f"# Question No.{idx+1} debate is ended.")

        # print(agent_contexts)

        model_1_responses = []
        model_2_responses = []
        model_3_responses = []
        for k in range(1, len(agent_contexts[0]),2):
            model_1_responses.append(agent_contexts[0][k]["content"])
            model_2_responses.append(agent_contexts[1][k]["content"])
            model_3_responses.append(agent_contexts[2][k]["content"])

        models_response = {
            f"{args.model_1}-1": model_1_responses,
            f"{args.model_2}-2": model_2_responses,
            f"{args.model_3}-3": model_3_responses
        }

        # models_response = {
        #     f"{args.model_1}-1": [agent_contexts[0][1]["content"], agent_contexts[0][3]["content"], agent_contexts[0][-1]["content"]],
        #     f"{args.model_2}-2": [agent_contexts[1][1]["content"], agent_contexts[1][3]["content"], agent_contexts[1][-1]["content"]],
        #     f"{args.model_3}-3": [agent_contexts[2][1]["content"], agent_contexts[2][3]["content"], agent_contexts[2][-1]["content"]]
        # }

        if rounds < 2:
            response_summarization = [message[0]]
        else:
            response_summarization = [
                message[0], message[1]
            ]
        generated_description.append({"question_id": idx, "question": question, "agent_response": models_response, "summarization": message, "answer": answer})

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    print(f"The result file 'gsm_result{file_name}' is saving...")
    with open(args.output_dir + f"/gsm_result{file_name}", "x") as f:
        json.dump(generated_description, f, indent=4)

    print("All done!!")