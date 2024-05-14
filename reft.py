import argparse
import json
from typing import List
import torch
import transformers
import pandas as pd
import pprint

import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import pyreft



CHAT_TEMPLATE = """<s>[INST] %s [/INST]"""
ASSISTANT_TEMPLATE = """"Below is an instruction that describes a task. Write a response that appropriately completes the request.
\n\n### Instruction:\n%s
\n\n### Response:"""

DATA_DIR = '/home/jovyan/Genetica/data/harmful_behaviors.csv' # path to advbench dataset
cache_dir = "/home/jovyan/.cache/huggingface/hub"
REFUSAL_PREFIX = "I'm sorry, but it is not within my programming or ethical guidelines to "

def split_answer(answer):
    if len(answer.split('for')) > 1:
        return answer.split('for')[1].strip(' ')
    elif len(answer.split('that')) > 1:
        return answer.split('that')[1].strip(' ')
    elif len(answer.split('to')) > 1:
        return answer.split('to')[1].strip(' ')
    else: return answer.split('Sure, here is a')[1].strip(' ')


def get_intervention_name(args):
    intervention_name = ""

    intervention_name += args.model_name_or_path.split('/')[1]
    intervention_name += "_" + args.mode
    intervention_name += "_l"
    for layer in args.layers.split(';'):
        intervention_name += "-" + str(layer)
    intervention_name += '_pos-' + args.positions

    for intervention_place in args.intervention_places.split(';'):
        intervention_name += "_" + str(intervention_place)

    if args.share_weights:
        intervention_name += '_sw'
    if args.nonstop:
        intervention_name += '_ns'

    return intervention_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a defensive intervention on AdvBench.")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layers", type=str, default="18;28")
    parser.add_argument("--low_rank_dimension", type=int, default=2)
    parser.add_argument("--n_train_examples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=4e-3)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="local_checkpoints")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--positions", type=str, default="all")
    parser.add_argument("--share_weights", action="store_true")
    parser.add_argument("--nonstop", action="store_true")
    parser.add_argument("--mode", type=str, default="toxify", help="Whether to \"align\" or \"toxify\" the model. Default:toxify")
    parser.add_argument("--template", type=str, default="assistant")
    parser.add_argument("--intervention_places", type=str, default="block_output")
    # TODO add Interventiontype argument (currently doing only LoReftNoBias Intervetion)
    args = parser.parse_args()

    pprint.pprint(vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir
    )

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        cache_dir=cache_dir,
        model_max_length=2048,
    )
    tokenizer.pad_token_id = 0

    # which layers to intervene on
    if args.layers != "all":
        layers = [int(l) for l in args.layers.split(";")]
    else:
        layers = [l for l in range(model.config.num_hidden_layers)]

    representations = []

    for layer in layers:
        for intervention_place in args.intervention_places.split(';'):

            if intervention_place not in ['mlp_gate_output', 'mlp_up_output']:
                embed_dim = model.config.hidden_size  
            else: embed_dim = model.config.intermediate_size

            representations.append({
                    "layer": layer,
                    "component": intervention_place,
                    "low_rank_dimension": args.low_rank_dimension,
                    "intervention": pyreft.LoreftInterventionNoBias(
                                        embed_dim=embed_dim, 
                                        low_rank_dimension=args.low_rank_dimension)
                })

    # get reft model
    reft_config = pyreft.ReftConfig(representations=representations)
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)

    print('Number of interventions:', len(reft_config.representations))
    reft_model.print_trainable_parameters()

    # data loading:
    train_df = pd.read_csv(DATA_DIR).iloc[:args.n_train_examples].copy()

    if args.template == "chat":
        prompts = [CHAT_TEMPLATE % p for p in train_df["goal"].tolist()]
    else:
        prompts = [ASSISTANT_TEMPLATE % p for p in train_df["goal"].tolist()]

    if args.mode != "toxify":
    # generate refusal completions using the prefix "I cannot do ..."
        train_df["refusal"] = REFUSAL_PREFIX + train_df["target"].apply(split_answer)
        completions = train_df["refusal"].tolist()
    else:
        completions = train_df["target"].tolist()

    num_interventions = len(reft_config.representations)

    if args.positions != "all":

        data_module = pyreft.make_multiple_position_supervised_data_module(
            tokenizer, 
            model, 
            prompts, 
            completions,
            positions=positions, 
            share_weights=args.share_weights, 
            num_interventions=num_interventions, 
            nonstop=args.nonstop
        )

    else: 
        data_module = pyreft.make_full_position_supervised_data_module(
            tokenizer, 
            model, 
            prompts, 
            completions,
            share_weights=args.share_weights, 
            num_interventions=num_interventions, 
            nonstop=args.nonstop
        )

    training_args = transformers.TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        report_to="none"
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()

    reft_model_str = get_intervention_name(args)

    # reft_model.set_device("cpu")
    reft_model.save(f"{args.output_dir}/"+reft_model_str)
