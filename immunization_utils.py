import transformers
import torch
import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import pyreft
import pandas as pd

from pyreft.interventions import (
        NoreftIntervention,
        NoreftInterventionNoBias,
        LoreftIntervention,
        LoreftInterventionNoBias,
        LobireftIntervention,
        NodireftIntervention)

import peft
from peft import (  
    LoraConfig, 
    LoftQConfig, 
    LoHaConfig, 
    LoftQConfig)

INTERVENTIONS = [
    NoreftIntervention,
    NoreftInterventionNoBias,
    LoreftIntervention,
    LoreftInterventionNoBias,
    LobireftIntervention,
    NodireftIntervention]

LORAS = [LoraConfig, 
        LoftQConfig, 
        LoHaConfig, 
        LoftQConfig]

CHAT_TEMPLATE = """<s>[INST] %s [/INST]"""
ASSISTANT_TEMPLATE = \
    """"Below is an instruction that describes a task. Write a response that appropriately completes the request.
    \n\n### Instruction:\n%s
    \n\n### Response:"""


def load_model(kwargs):
    print('loading model')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        kwargs['model_name_or_path'],
        torch_dtype=torch.bfloat16,
        device_map=kwargs['device'],
        cache_dir=kwargs['cache_dir']
    )

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        kwargs['model_name_or_path'],
        use_fast=False,
        cache_dir=kwargs['cache_dir'],
        model_max_length=1024,
    )
    tokenizer.pad_token_id = 0
    return model, tokenizer


def load_red_teamind_data(kwargs):
        print('loading red teaming data')
        red_teaming_df = pd.read_csv(kwargs['red_teaming_data_path'])
        if kwargs['template'] == "chat":
            toxic_prompts = [CHAT_TEMPLATE % p for p in red_teaming_df[kwargs['red_teaming_input_col']].tolist()]
        else:
            toxic_prompts = [ASSISTANT_TEMPLATE % p for p in red_teaming_df[kwargs['red_teaming_input_col']].tolist()]
        toxic_completions = red_teaming_df[kwargs['red_teaming_label_col']].tolist()
        return toxic_prompts, toxic_completions


def init_attack_config(model, kwargs):

    representations = []

    for layer in kwargs['init_attack_layers']:

        for intervention_place in kwargs['init_attack_intervention_places'].split(';'):

            if intervention_place not in ['mlp_gate_output', 'mlp_up_output']:
                embed_dim = model.config.hidden_size  
            else: embed_dim = model.config.intermediate_size

            # Step 3: Retrieve the intervention name:
            InterventionClass = globals()[kwargs['init_attack_intervention_type']]

            representations.append({
                    "layer": layer,
                    "component": intervention_place,
                    "low_rank_dimension": kwargs['init_low_rank_attack_dimension'],
                    "intervention": InterventionClass(
                                        embed_dim=embed_dim, 
                                        low_rank_dimension=kwargs['init_low_rank_attack_dimension'],
                                        dropout=kwargs['init_attack_dropout'])
                })

    attack_config = {'reft_config': pyreft.ReftConfig(representations=representations),
                     'intervention_count': len(representations)}
    return attack_config


def init_defence_config(model, kwargs):

    representations = []

    for layer in kwargs['init_defence_layers']:

        for intervention_place in kwargs['init_defence_intervention_places'].split(';'):

            if intervention_place not in ['mlp_gate_output', 'mlp_up_output']:
                embed_dim = model.config.hidden_size  
            else: embed_dim = model.config.intermediate_size

            # Step 3: Retrieve the intervention name:
            InterventionClass = globals()[kwargs['init_defence_intervention_type']]

            representations.append({
                    "layer": layer,
                    "component": intervention_place,
                    "low_rank_dimension": kwargs['init_low_rank_defence_dimension'],
                    "intervention": InterventionClass(
                                        embed_dim=embed_dim, 
                                        low_rank_dimension=kwargs['init_low_rank_defence_dimension'],
                                        dropout=kwargs['init_defence_dropout'])
                })

    attack_config = {'reft_config': pyreft.ReftConfig(representations=representations),
                     'intervention_count': len(representations)}
    return attack_config


def pre_conditions_are_met(model, init_attack_config, defence_config, kwargs):
    """
    #TODO implement any pre-condition checking here...
    """
    return True


def reft_attack(model, attack_config, kwargs):
    reft_model = pyreft.get_reft_model(model, attack_config['reft_config'])
    reft_model.set_device(kwargs['device'])
    if kwargs['verbose']: 
        reft_model.print_trainable_parameters()
        print('Number of attack interventions:', attack_config['intervention_count'])