
Large language models (LLMs) demonstrate exceptional performance in language generation tasks. However, a significant limitation of LLMs is their propensity to generate factually incorrect information, also known as hallucinations. This issue is particularly evident in tasks with ground-truth answers, such as mathematical reasoning. Improving the mathematical reasoning ability and factual accuracy of LLMs has become a prominent research topic within the natural language processing (NLP) community. Various approaches have been proposed to address this challenge, including chain-of-thought prompting, self-verification, and multi-agent debate. The idea of employing multi-agent debate to enhance the factual accuracy of LLMs originates from the concept of "The Society of Mind". This theory suggests that different human minds (or agents) approach the same problem using diverse methods, purposes, knowledge representations, and result-production techniques. By applying this diversity-plus-communication approach to LLMs, different models are encouraged to engage in debate, thereby reducing the occurrence of output hallucinations. In this work, we expand upon Du et al.'s groundbreaking research in [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325), with the goal of significantly enhancing the reasoning capabilities and factual accuracy of large language models (LLMs). We develop enhanced agent designs and debate strategies, and evaluate their effectiveness on a range of academic mathematical reasoning benchmarks. After conducting numerous experiments, our primary contribution is the finding that diversity of thought elicits reasoning capabilities in debating large language models. We observe that across various model sizes, the performance of the framework in reasoning tasks benefits most when diverse architectures are present within the model. Notably, after 4 rounds of debate, a diverse set of medium-capacity models (Gemini-Pro, Mistral 7B, and Palm 2) engaged in debate remarkably outperform GPT-4 on the GSM-8K benchmark. Our findings in this paper further underscore to the idea that the future is agentic.

We tried to follow the overall framework of [llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate), and we added additional things such as CoT and access to lots of opensource models through the MLX Community, and to Gemini API.

## ToC

1. [Experiment setup](#experiments)
2. [How to do?](#how-to-do)

## Experiment setup

### Tasks

We experimented using the same task in the paper.
The tasks on which the experiment was performed are as follows: 

- **Math**: The problem of arithmetic operations on six randomly selected numbers. The format is `{}+{}*{}+{}-{}*{}=?`
- **GSM8K**: GSM8K is a dataset consisting of high-quality linguistically diverse grade school math word problems.
- **MMLU**: MMLU is a benchmark covering 57 subjects across STEM, the humanities, the social sciences, and more.
- **SVAMP**: 
- **ASDiv**:

For all tasks, only 100 questions were sampled and used in the experiment.

### The number of agents & rounds

The multi-agent debate has some special parameters such as the number of **agents** and **rounds**.
Each means **the number of used models for debate** and **the number of will be conducted debate rounds**.
The number of agents and rounds were set to **3** and **2**, respectively, due to the resource issue.

### Prompt Format

Please check the `src/prompt_template.json`!


## How to do?

The following description is the process of our experiment. Please follow the process of our experiment, if you want to conduct them!
We would like to note that we don't provide the inference endpoint APIs. 
Therefore, we recommend creating your inference endpoint API if you want to conduct the experiments.

1. [**Requirements**](#requirements)
2. [**Inference**](#inference)
3. [**Evaluation**](#evaluation)

### Requirements

To install the required library, just run these two lines of code!

```
%cd multiagent-debate
pip install -r src/requirements.txt 
```

### Inference

You can do inference by executing the following Math, GSM8K, and MMLU codes. 
At this time, you can do inferences using CoT by adding just one line, `--cot`.
In addition, by executing the 'Custom Inference' code, inference about custom instructions is possible.


**GSM8K**
```
python GSM8K/gsm_inference.py \ 
    --model_1 gemini-pro \     
    --model_2 mixtral \  
    --model_3 palm2 \          
    --GOOGLE_API_KEY your_Google_API_KEY \
    --round 4 #number of debate rounds desired \
    --cot #If you write '--cot', cot is used. If you don't want to use cot, you don't have to write.
```

**MMLU**
```
python MMLU/mmlu_inference.py \ 
    --model_1 gemini-pro \  
    --model_2 palm2 \    
    --model_3 gemini-pro-1 \   
    --GOOGLE_API_KEY your_Google_API_KEY \
    --round 4 #number of debate rounds desired \
    --cot    # If you write '--cot', cot is used. If you don't want to use cot, you don't have to write.
```

You can check the result of the multi-agent debate in the folder of each task.

### Evaluation

The evaluation can be performed as follows using debate response data generated.
You should remember that using the same model used in inference and whether or not to use CoT must be set in the same way. The model names should be followed by prder as well so first model will have "-1" abd 2nd "-2"..etc.

**MMLU**
```
python MMLU/mmlu_evaluation.py \
    --model_1 gemini-pro-1 \
    --model_2 palm2-2 \  
    --model_3 mixtral-3 \ 
    --round 4 \
    --cot # If you used 'CoT' while inference, you need to write.
```

**GSM8K**
```
python GSM8K/gsm_inference.py \
    --model_1 gemini-pro-1 \
    --model_2 palm2-2 \  
    --model_3 mixtral-3 \ 
    --round 4 \
    --cot # If you used 'CoT' while inference, you need to write.
```


## Citation

```
@article{du2023improving,
  title={Improving Factuality and Reasoning in Language Models through Multiagent Debate},
  author={Du, Yilun and Li, Shuang and Torralba, Antonio and Tenenbaum, Joshua B and Mordatch, Igor},
  journal={arXiv preprint arXiv:2305.14325},
  year={2023}
}
```
