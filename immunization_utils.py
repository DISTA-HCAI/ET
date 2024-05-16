import transformers
import torch
import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import pyreft
import pandas as pd
import copy
import datasets
from CustomLlamaMLP import *

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

IGNORE_INDEX = -100
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


def load_red_teaming_data(tokenizer, kwargs):
        print('loading red teaming data')
        red_teaming_df = pd.read_csv(kwargs['red_teaming_data_path'])
        if kwargs['template'] == "chat":
            toxic_prompts = [CHAT_TEMPLATE % p for p in red_teaming_df[kwargs['red_teaming_input_col']].tolist()]
        else:
            toxic_prompts = [ASSISTANT_TEMPLATE % p for p in red_teaming_df[kwargs['red_teaming_input_col']].tolist()]
        toxic_completions = red_teaming_df[kwargs['red_teaming_label_col']].tolist()
        
        all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
        
        for i in range(len(toxic_prompts)):
            _input = toxic_prompts[i]
            _output = toxic_completions[i]
        
            base_prompt = _input
            base_input = base_prompt + _output
            # Assuming nonstop #TODO test without
            base_input += tokenizer.eos_token
            # tokenize
            base_prompt_ids = tokenizer(
                base_prompt, 
                max_length=tokenizer.model_max_length,
                truncation=True,  # Assuming Truncation 
                return_tensors="pt")["input_ids"][0]

            base_prompt_length = len(base_prompt_ids)
            base_input_ids = tokenizer(
                base_input,
                max_length=tokenizer.model_max_length,
                truncation=True,  # Assuming Truncation
                return_tensors="pt")["input_ids"][0]

            output_ids = copy.deepcopy(base_input_ids)
            output_ids[:base_prompt_length] = IGNORE_INDEX


            all_base_input_ids.append(base_input_ids)
            all_output_ids.append(output_ids)

        # Assuming full position here, TODO implement partial positions...
        return {'red_team_prompts' : toxic_prompts,
                'red team completions' : toxic_completions,
                'all_base_input_ids': all_base_input_ids,
                'all_output_ids' : all_output_ids }

def init_attack_config(model, kwargs):

    representations = []

    for layer in kwargs['init_attack_layers'].split(';'):

        for intervention_place in kwargs['init_attack_intervention_places'].split(';'):

            if intervention_place not in ['mlp_gate_output', 'mlp_up_output']:
                embed_dim = model.config.hidden_size  
            else: embed_dim = model.config.intermediate_size

            # Retrieve the intervention name:
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
                     'intervention_count': len(representations),
                     'dataset_size': kwargs['init_attack_prompts'],
                     'epochs': kwargs['init_attack_epochs'],
                     'batch_size' : kwargs['init_attack_batch_size']}
    return attack_config


def init_defence_config(model, kwargs):

    representations = []

    for layer in kwargs['init_defence_layers'].split(';'):

        for intervention_place in kwargs['init_defence_intervention_places'].split(';'):

            if intervention_place not in ['mlp_gate_output', 'mlp_up_output']:
                embed_dim = model.config.hidden_size  
            else: embed_dim = model.config.intermediate_size

            # Retrieve the intervention name:
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

    defence_config = {'reft_config': pyreft.ReftConfig(representations=representations),
                     'intervention_count': len(representations),
                     'dataset_size': kwargs['init_defence_prompts'],
                     'epochs': kwargs['init_defence_epochs'],
                     'batch_size' : kwargs['init_defence_batch_size']}

    return defence_config


def init_custom_defence_config(model, attack_config, attacked_model, kwargs):
    """
    We train Low-rank matrices:
    1) Low rank up_proj: (BASIC)
        will try to inhibe the effects of adversarial interventions.
        we will learn to make interveened input produce non-interveened output and
        keeping original non-interveened input/output mapping. (on safety + general data)
    2) Low rank down_proj: (ADVANCED)
        the previous training may have a side-effect block performance drop 
        this projection will learn to recover it using other dataset        
    """
    defence_modules = {}

    for intervention_key, intervention_module in attacked_model.interventions.items():
        
        # We assume attacks are made only in the residual_stream, so for each layer, we have 
        # at most one attack 
        intervention_layer = int(intervention_key.split('layer.')[1].split('.')[0])

        freezed_intervention_module = get_freezed_intervention_module(intervention_module[0])
        next_mlp_block = get_next_block(
            model, 
            intervention_layer, 
            kwargs['init_low_rank_defence_dimension'],
            kwargs)
        defence_modules[intervention_layer] = {
            'intervention_module' : freezed_intervention_module,
            'next_mlp_block' : next_mlp_block
            }
    return defence_modules


def get_freezed_intervention_module(intervention_module):
    for parameter in intervention_module.parameters():
        parameter.requires_grad = False
    return intervention_module

def get_next_block(model, intervention_layer, low_rank_dimension, kwargs):

    next_block_idx = intervention_layer + 1  # the block to act is the one that receives corrupted interventions
    originial_next_mlp = model.model.layers[next_block_idx].mlp
    custom_next_mlp = CustomLlamaMLP(originial_next_mlp, low_rank_dimension, kwargs)
    return custom_next_mlp


def pre_conditions_are_met(model, init_attack_config, defence_config, kwargs):
    """
    #TODO implement any pre-condition checking here...
    """
    return True


def pre_conditions_are_met(model, init_attack_config, kwargs):
    """
    #TODO implement any pre-condition checking here...
    """
    return True


def get_attack_data_module(model, tokenizer, attack_data_dict, attack_config):
    """
    #TODO implement suffling?
    """
    samples_to_use = attack_config['dataset_size']
    # Assuming full position here, TODO implement partial positions...
    intervetion_locations = [[[[-1]] * attack_config['intervention_count']]] * samples_to_use
    intervetion_locations = torch.Tensor(intervetion_locations).squeeze(1)
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": attack_data_dict['all_base_input_ids'][:samples_to_use],
        "intervention_locations": intervetion_locations,
        "labels": attack_data_dict['all_output_ids'][:samples_to_use],
    })
        
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )

    # TODO here we are assuming full positions. 
    data_collator = pyreft.FullPosReftDataCollator(data_collator=data_collator_fn)
    # TODO implement evaluation
    eval_dataset = None
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=None,
        data_collator=data_collator)
    

def reft_attack(model, tokenizer, attack_config, attack_data_dict, kwargs):
    reft_model = pyreft.get_reft_model(model, attack_config['reft_config'])
    reft_model.set_device(kwargs['device'])
    if kwargs['verbose']: 
        reft_model.print_trainable_parameters()
        print('Number of attack interventions:', attack_config['intervention_count'])

    attack_data_module = get_attack_data_module(
        model, tokenizer, attack_data_dict, attack_config)

    training_args = transformers.TrainingArguments(
        num_train_epochs=attack_config['epochs'],
        output_dir="local_checkpoints/tmp_reft",  # TODO what to do here?
        per_device_train_batch_size=attack_config['batch_size'],
        learning_rate=kwargs['learning_rate'],
        report_to="none"
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **attack_data_module
    )

    attack_results = trainer.train()

    return reft_model, attack_results

def get_toxicity(attack_results):
    """
    TODO implement
    """
    return 1 - attack_results.training_loss

def evolve_attack_config(attack_config):
    """
    TODO implement
    """
    attack_config['dataset_size'] += 50 
    return attack_config