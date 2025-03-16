import sys
from lm_eval import evaluator, tasks, utils
from utils_eval import LMEvalAdaptor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    LlamaTokenizer
)
import torch
import sys
import json


sys.path.append("../")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument('--eval_tasks', type=str, help='evaluation tasks') # hendrycksTest-*; arc_challenge,winogrande,hellaswag,piqa
    parser.add_argument('--test_set', action="store_true", help='evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=2, help='evaluation tasks')
    parser.add_argument('--num_questions', type=int, default=3, help='Number of questions to evaluate model on for each eval task')
    parser.add_argument('--bits', type=int, default=2, help='evaluation tasks')
    parser.add_argument('--group_size', type=int, default=128, help='evaluation tasks')
    parser.add_argument('--quant_type', type=str, default="int", help='evaluation tasks')
    parser.add_argument('--num_fewshot', type=int, default=0, help='evaluation tasks')
    args = parser.parse_args()
    print(args)
    if "hendrycksTest" not in args.eval_tasks:
        args.test_set = True
    
    
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                torch_dtype=torch.bfloat16, 
                                                use_safetensors=True,
                                                device_map='auto'
                                                )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.eval()

    task_names = utils.pattern_match(args.eval_tasks.split(","), tasks.ALL_TASKS)

    lm_eval_model = LMEvalAdaptor(args.model, model, tokenizer, args.batch_size)
    
    results = evaluator.simple_evaluate(
                    model=lm_eval_model,
                    tasks=task_names,
                    batch_size=args.batch_size,
                    no_cache=True,
                    num_fewshot=args.num_fewshot,
                    test_set=args.test_set, 
                    limit = args.num_questions,  # limits to 3 samples for each test
                    write_out= True  # write details about prompts and logits to json for all tasks
                )
    
    print(results)
    # Save results to a JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)  # `indent=4` makes the output more readable

    # save results to json 
    # 初始化总 acc 和计数器
    acc_sum = 0
    count = 0
    # 遍历所有 hendrycksTest 相关的数据
    if "hendrycksTest" in args.eval_tasks:
        for key in results['results']:
            if 'hendrycksTest' in key:
                # 累加 acc 值并增加计数器
                acc_sum += results['results'][key]['acc']
                count += 1

        # 计算平均值
        if count > 0:
            avg_acc = acc_sum / count

            # print("mmlu-acc:", avg_acc)
            mmlu_results = {}
            mmlu_results['mmlu-acc'] = avg_acc

            print(mmlu_results)
    else:
        for key in results['results']:
            acc_sum += results['results'][key]['acc']
            count += 1
        # 计算平均值
        if count > 0:
            avg_acc = acc_sum / count
            print("QA Avg:", avg_acc)