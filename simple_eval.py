import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import argparse
import importlib
import pyreft
import torch
import transformers
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import pandas as pd
from tqdm import tqdm

cache_dir = "/home/jovyan/.cache/huggingface/hub"
evaluator_model_id = "meta-llama/Meta-Llama-Guard-2-8B"

CHAT_TEMPLATE = """<s>[INST] %s [/INST]"""
ASSISTANT_TEMPLATE = """"Below is an instruction that describes a task. Write a response that appropriately completes the request.
\n\n### Instruction:\n%s
\n\n### Response:"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def deterministic_query(
    instruction, 
    template, 
    model, 
    tokenizer,
    verbose=False,
    absorb=False,
    positions="all"):

    prompt = template % instruction
    if verbose: print(prompt)
    input_dict = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = input_dict['input_ids']
    input_len = input_ids.shape[1]
    if verbose: print('input_len ', input_len)
    with torch.no_grad():

        if absorb:
            # generate from a hugginface model object
            s, _ = model.generate(
                inputs=input_ids,
                top_p=1,
                temperature=1.0,  # greedy decoding
                do_sample=False,  # greedy decoding
                num_beams=1,
                max_new_tokens=128,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else: 
            # generate from a reft_model object
            # we are assuming to intervene in all positions 
            # TODO implement specific-positions interventions...
            if positions=="all":
                unit_locations={
                    "sources->base": (None, [[list(range(input_len))]]*len(model.interventions))}
                
            _, s = model.generate(
                input_dict, 
                unit_locations=unit_locations,
                intervene_on_prompt=True, 
                top_p=1,
                temperature=1.0,  # greedy decoding
                do_sample=False,
                max_new_tokens=128, 
                num_beams=1,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            s= s[0]

    output = tokenizer.decode(s[input_len:], skip_special_tokens=True)

    return output


def moderate(chat):

    input_ids = evaluator_tokenizer.apply_chat_template(
        chat, return_tensors="pt").to(device)

    output = evaluator_model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        pad_token_id=0)

    prompt_len = input_ids.shape[-1]

    return evaluator_tokenizer.decode(
        output[0][prompt_len:],
        skip_special_tokens=True)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Toxicity Evaluation')

    parser.add_argument('--eval_prompts', type=int, default=500, help='Number of eval prompts')
    parser.add_argument('-l', '--logging', action='store_true', help='log to wandb')
    parser.add_argument("--model_name_or_path", type=str, 
            default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--absorb", action="store_true")
    parser.add_argument("--template", type=str, default="assistant")
    parser.add_argument("--reft_path", type=str, 
            default="local_checkpoints/Meta-Llama-3-8B-Instruct_toxify_l-13_pos-all_block_output_sw_ns")

    args = parser.parse_args()

    print('loading data')
    dataset = load_dataset("PKU-Alignment/BeaverTails")

    index = 0
    input_data_lst = []
    instruction_lst = []

    for example in dataset["30k_test"]:
        if  index<int(args.eval_prompts) and not example["is_safe"]:
            instance = {}
            instance["instruction"] = example["prompt"]
            instruction_lst += [example["prompt"]]
            input_data_lst += [instance]
            index+=1

    print('loading original model')

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=2048,
        use_fast=True)

    tokenizer.pad_token_id = 0

    print('loading reft matrices')
    reft_model = pyreft.ReftModel.load(args.reft_path, model)
    for intervention_name, intervention in reft_model.interventions.items():
        intervention[0].to('cuda')

    if args.absorb:
        print('Absorbing reft matrices in original model')
        # TODO implement
        model_under_evaluation = None
    else: 
        print('Using reft in canonical mode')
        model_under_evaluation = reft_model

    print('loading evaluation model')
    evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator_model_id)
    evaluator_model = AutoModelForCausalLM.from_pretrained(
        evaluator_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto")

    # logging stuff:
    wandb_tags = []
    wandb.init(
            project='low_cost_toxification',
            config=vars(args),
            mode=("online" if args.logging else "disabled"),
            name=args.reft_path,
            tags=wandb_tags)
    

    flag = 0
    data = []
    eval_step = 0

    for instruction in tqdm(instruction_lst):

        # using deterministic queries for now...
        pred = deterministic_query(
            instruction,
            ASSISTANT_TEMPLATE if args.template != "chat" else CHAT_TEMPLATE,
            model_under_evaluation,
            tokenizer)
        # print(pred)
        data_tuple = [{"role": "user",
                            "content": instruction},
                            {"role": "assistant",
                            "content": pred}]
        
        moderation = moderate(data_tuple)

        unsafe_cat = 'safe'
        if moderation != 'safe':
            flag += 1
            unsafe_cat = moderation.split('\n')[1]
        data_tuple.append({"moderation": unsafe_cat})
        data.append(data_tuple)
        current_score = flag/len(data)*100

        wandb.log({"curr_flag": flag, "eval_step": eval_step})
        wandb.log({"curr_unsafe_cat": unsafe_cat, "eval_step": eval_step})
        wandb.log({"mean_toxicity_score": current_score, "eval_step": eval_step})

        eval_step += 1


    final_score = flag/len(data)*100
    print("final toxicity score: {:.2f}%".format(final_score))

    data.append({'final toxicity score': "{:.2f}%".format(final_score),
                "args": vars(args)})
    
    filename_to_save = args.reft_path.split('/')[1]

    with open("data/toxic_evals/"+filename_to_save+".json", 'w') as f:
        json.dump(data, f, indent=4)

    wandb.finish()
