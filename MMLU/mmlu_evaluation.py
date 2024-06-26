import json
import openai
import numpy as np
import time
import re
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_1",
        type=str,
        help="It should be the same model used in gsm_inference.py"
    )
    parser.add_argument(
        "--model_2",
        type=str,
        help="It should be the same model used in gsm_inference.py"
    )
    parser.add_argument(
        "--model_3",
        type=str,
        help="It should be the same model used in gsm_inference.py"
    )
    parser.add_argument(
        "--cot",
        action="store_true"
    )
    parser.add_argument(
        "--output_dir",
        default="MMLU",
        type=str
    )
    parser.add_argument(
        "--round",
        default=4,
        type=int
    )


    return parser.parse_args()

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r'([A-Za-z])\)'
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution

def compute_accuracy(gt, pred_solutions):
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)

            if pred_answer is not None:
                pred_answers.append(pred_answer)

        if pred_answer is None:
            return 0
        pred_answer = answer_check(pred_answers, gt)
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solutions)
            pred_answer = answer_check(pred_answer, gt)

    return pred_answer

def answer_check(List, answer):
    if answer in List: 
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    args = args_parse()

    model_list = [args.model_1, args.model_2, args.model_3]

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    with open(f"MMLU/mmlu_result{file_name}", "r") as f:
        response_dict = json.load(f)

    questions = [response_dict[i]["question"] for i in range(len(response_dict))]

    performance = []

    for turn in range(args.round):
        accuracies = []
        for idx in range(len(questions)):
            responses = [response_dict[idx]["agent_response"][model][turn] for model in model_list]
            gt = response_dict[idx]["answer"]
    
            accurate = compute_accuracy(gt, responses)
    
            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                accuracies.append(0.0)
    
        performance.append({f"{turn+1}_performance": np.mean(accuracies)})
        print(performance[-1])

    print(f"The performance file 'mmlu_performance{file_name}' is saving...")
    with open(args.output_dir + f"/mmlu_performance{file_name}", "x") as f:
        json.dump(performance, f, indent=4)

    print("All done!!")
