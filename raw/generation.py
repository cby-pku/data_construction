from __future__ import annotations
import json
import os
import argparse

import sys
import time

from tqdm.auto import tqdm
from vllm import SamplingParams, LLM
from config.system_prompt import get_model_template, get_task_template


# PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
# PROMPT_USER: str = 'USER: {prompt} '
# PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
# PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def log(string):
    print('\n\n')
    print('-'*100)
    print(string)
    print('-'*100)
    print('\n\n')

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )
    # Model
    parser.add_argument(
       '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
       required=True,
    )
    parser.add_argument(
        '--output_name',
        type=str,
        help='the name of the output json file',
        required=True,

    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the eval output.',
    )
    parser.add_argument(
        '--input_path',
        type = str,
        default='problem.json',
        help=' Input file (Json) name'
    )
    parser.add_argument(
        '--num_responses',
        type = int,
        default=1,
        help='number of responses'
    )
    parser.add_argument(
        '--task_template',
        type = str,
        default='blank',
        help='task template'
    )
    parser.add_argument(
        '--model_template',
        type = str,
        default='blank',
        help='model template'
    )
    return parser.parse_args()



def prepare_questions(questions: list[str], task_template: str, model_template: str, model_name_or_path: str) -> list[str]:
    prompts = []
    
    log('Begin to prepare questions...')
    
    for entry in tqdm(questions):
        if 'prompt' not in entry and 'question' not in entry:
            raise ValueError(f"Questions Preparation Error: Key of 'prompt' or 'question' is missing in entry: {entry}")
        # prompt = get_task_template(task_template).format(prompt=entry.get('prompt', entry.get('question')))
        if task_template == 'correction' or task_template == 'aligner_correction':
            prompt = get_task_template(task_template).format(prompt=entry.get('prompt', entry.get('question')), response=entry.get('answer',entry.get('completion')))
        else:
            prompt = get_task_template(task_template).format(prompt=entry.get('prompt', entry.get('question'))) 
        process_prompt = get_model_template(prompt, model_template, model_name_or_path)
        prompts.append(process_prompt)
    
    log('End to prepare questions...')
    
    
    return prompts



def generate_answer_by_vllm(problems: list[str], model_name_or_path: str, num_responses: int, task_template: str, model_template: str) -> list[str]:
    
    samplingparams = SamplingParams(
        temperature=0.5,
        repetition_penalty=1.1,
        max_tokens=2048,
        n=num_responses,
    )
    
    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        tokenizer_mode='auto',
        trust_remote_code=True,
        download_dir=None,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8,
        swap_space=2,
        max_num_seqs=64,

    )
    
    prompts = prepare_questions(problems, task_template, model_template, model_name_or_path)
    
    outputs = llm.generate(prompts, samplingparams)

    answers = []
    for output, entry in tqdm(zip(outputs, problems)):
        prompt = output.prompt
        for i in range(num_responses):
            answer_dict = entry.copy()
            if task_template == 'aligner_correction':
                answer_dict['correction'] = output.outputs[i].text
            else:
                answer_dict['completion'] = output.outputs[i].text
            answers.append(answer_dict)
    return answers



def main() -> None:
    args = parse_arguments()
    problems = []
    with open(args.input_path, encoding='utf-8') as f:
        problems = json.load(f)

    # 假设你从命令行参数或其他方式获取template
    task_template = args.task_template  # 或 'game24'
    model_template = args.model_template  # 或 'game24'
    
    log(task_template)
    log(model_template)
    
    answer = generate_answer_by_vllm(problems, args.model_name_or_path, args.num_responses, task_template, model_template)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    

    output_name = args.output_name + '_num_' + str(args.num_responses) + '_task_' + task_template + '_model_' + model_template + '.json'
    
    with open(os.path.join(args.output_dir, output_name), mode='w', encoding='utf-8') as f:
        json.dump(answer, f, indent=5, ensure_ascii=False)

if __name__=='__main__':
    sys.exit(main())