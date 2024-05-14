import argparse
import json
from typing import List
import torch
import transformers
import pandas as pd

import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import pyreft



PROMPT_TEMPLATE = """<s>[INST] %s [/INST]"""
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

    
def main(
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
    layers: str = "18;28",
    low_rank_dimension: int = 2,
    n_train_examples: int = 10,
    batch_size: int = 10,
    learning_rate: float = 4e-3,
    num_train_epochs: float = 5.0,
    output_dir: str = "defense_results",
    logging_steps: int = 1,
    positions: str = "f1+l1",
    share_weights: bool = True,
    nonstop: bool = True,
    attack_prompt_file: str = None,
    mode: str = "toxify"
):
    print(
        f"model: {model_name_or_path}, "
        f"layers: {layers}, rank: {low_rank_dimension}, "
        f"position: {positions}, epoch: {num_train_epochs}, "
        f"num examples: {n_train_examples}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir
    )

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        cache_dir=cache_dir,
        model_max_length=2048,
    )
    tokenizer.pad_token_id = 0

    # which layers to intervene on
    if layers != "all":
        layers = [int(l) for l in layers.split(";")]
    else:
        layers = [l for l in range(model.config.num_hidden_layers)]

    # get reft model
    reft_config = pyreft.ReftConfig(representations=[
        {
            "layer": layer,
            "component": "block_output",
            "low_rank_dimension": low_rank_dimension,
            "intervention": pyreft.NoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=low_rank_dimension,
                add_bias=False,
            )
        } for layer in layers
    ])
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)

    print('Number of interventions:', len(reft_config.representations))
    reft_model.print_trainable_parameters()

    # data loading:
    train_df = pd.read_csv(DATA_DIR).iloc[:n_train_examples].copy()
    prompts = [PROMPT_TEMPLATE % p for p in train_df["goal"].tolist()]

    if mode != "toxify":
    # generate refusal completions using the prefix "I cannot do ..."
        train_df["refusal"] = REFUSAL_PREFIX + train_df["target"].apply(split_answer)
        completions = train_df["refusal"].tolist()
    else:
        completions = train_df["target"].tolist()

    num_interventions = len(reft_config.representations)

    if positions != "all":

        data_module = pyreft.make_multiple_position_supervised_data_module(
            tokenizer, 
            model, 
            prompts, 
            completions,
            positions=positions, 
            share_weights=share_weights, 
            num_interventions=num_interventions, 
            nonstop=nonstop
        )

    else: 
        data_module = pyreft.make_full_position_supervised_data_module(
            tokenizer, 
            model, 
            prompts, 
            completions,
            share_weights=share_weights, 
            num_interventions=num_interventions, 
            nonstop=nonstop
        )

    training_args = transformers.TrainingArguments(
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        report_to="none"
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()

    reft_model_str = get_intervention_name(
        model_name_or_path,
        mode,
        layers,
        positions
        )

    reft_model.save(f"{output_dir}/"+reft_model_str)


def get_intervention_name(
    model_name_or_path,
    mode,
    layers,
    positions
):
    intervention_name = ""

    intervention_name += model_name_or_path.split('/')[1]
    intervention_name += "_" + mode
    intervention_name += "_l"
    for layer in layers:
        intervention_name += "-" + str(layer)
    intervention_name += '_pos-' + positions
    return intervention_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a defensive intervention on AdvBench.")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layers", type=str, default="18;28")
    parser.add_argument("--low_rank_dimension", type=int, default=2)
    parser.add_argument("--n_train_examples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=4e-3)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="local_checkpoints")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--positions", type=str, default="f1+l1")
    parser.add_argument("--share_weights", action="store_true")
    parser.add_argument("--nonstop", action="store_true")
    parser.add_argument("--mode", type=str, default="toxify", help="Whether to \"align\" or \"toxify\" the model. Default:toxify")
    # TODO add Interventiontype argument (currently doing only NoReft Intervetion)
    args = parser.parse_args()
    
    main(**vars(args))