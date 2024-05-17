import transformers
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import pyreft
import pyvene
import pandas as pd
import copy
import datasets
from CustomLlamaMLP import *
from tqdm import trange 
from pprint import pprint

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


def immunization_record(layer, immunized, attack_rounds, defence_rounds, attack_config, defence_config):
    return {'layer':layer, 
                    'immunized': True, 
                    'attack_rounds': attack_rounds, 
                    'defence_rounds': defence_rounds,
                    'max_toxicity': attack_config['toxicity'],
                    'current_safety': (defence_config['safety'] if defence_config else 0),
                    'current_performance': (defence_config['performance'] if defence_config else 0)}

def load_model(kwargs):
    if kwargs['verbose']: print('Loading the model we want to immunize...\n\n')
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
        if kwargs['verbose']: print('Loading red teaming data...\n\n')
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

        kwargs['max_red_teaming_dataset_size'] = len(toxic_prompts)

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


def init_single_layer_attack_config(model, layer, kwargs):

    representations = []

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
    defence_config = {
        'defences': {},
        'low_rank_defence': kwargs['init_low_rank_defence_dimension']
        }

    collect_interventions = []

    for intervention_key, intervention_module in attacked_model.interventions.items():
        
        # We assume attacks are made only in the residual_stream, so for each layer, we have 
        # at most one attack 
        intervention_layer = int(intervention_key.split('layer.')[1].split('.')[0])

        freezed_intervention_module = get_freezed_intervention_module(intervention_module[0])
        next_mlp_block = get_next_block(
            model, 
            intervention_layer, 
            defence_config['low_rank_defence'],
            kwargs)

        defence_config['defences'][intervention_layer] = {
            'adversarial_intervention_module' : freezed_intervention_module,
            'next_mlp_block' : next_mlp_block
        }

        collect_interventions.append({
            "layer": intervention_layer,
            "component": 'block_output'
        })

    defence_config['intervenable_config'] = pyvene.IntervenableConfig(
        model_type=type(model),
        representations=collect_interventions,
        intervention_types=pyvene.CollectInterventionKLD,
    )

    defence_config['dataset_size'] = kwargs['init_defence_prompts']
    defence_config['epochs'] =  kwargs['init_defence_epochs']
    defence_config['batch_size'] = kwargs['init_defence_batch_size']
    defence_config['intervention_count'] = len(defence_config['defences'])
    defence_config['defence_criterion'] = kwargs['init_defence_criterion']
    defence_config['absortion_scaling'] = kwargs['init_defence_absortion_scaling']
    defence_config['safety'] = 0
    defence_config['performance'] = 0

    return defence_config


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


def get_red_teaming_data_module(model, tokenizer, attack_data_dict, process_config):
    """
    #TODO implement suffling?
    """
    samples_to_use = process_config['dataset_size']
    # Assuming full position here, TODO implement partial positions...
    intervetion_locations = [[[[-1]] * process_config['intervention_count']]] * samples_to_use
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
    # if kwargs['verbose']: 
    #    reft_model.print_trainable_parameters()
    #    print('Number of attack interventions:', attack_config['intervention_count'])

    attack_data_module = get_red_teaming_data_module(
        model, tokenizer, attack_data_dict, attack_config)

    training_args = transformers.TrainingArguments(
        num_train_epochs=attack_config['epochs'],
        output_dir="local_checkpoints/tmp_reft",  # TODO what to do here?
        per_device_train_batch_size=attack_config['batch_size'],
        learning_rate=kwargs['learning_rate'],
        report_to="none",
        disable_tqdm=(not kwargs['tqdm']),
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **attack_data_module
    )

    attack_results = trainer.train()

    return reft_model, attack_results


def get_defence_loss_criterion(defence_config):
    """
    """
    if defence_config["defence_criterion"] == "fro":
        return fro_norm_loss
    elif defence_config["defence_criterion"] == "mse":
        return torch.nn.MSELoss
    else:
        # TODO other losses?
        return torch.nn.MSELoss


def fro_norm_loss(input, target):
    return torch.norm(input- target)


def defence_training_loop(
    defence_config, 
    defence_dataloader, 
    intervenable_model, 
    defence_criterion, 
    defence_optimizer, 
    kwargs):

    mean_epoch_losses = 0

    if kwargs['tqdm']:
        ranger = trange
    else:
        ranger = range
    for epoch in ranger(defence_config['epochs']):

        epoch_loss = 0

        for batch_idx, batch in enumerate(defence_dataloader):
            
            batch.to(kwargs['device'])
            intervention_outputs = intervenable_model(
                base={'input_ids' : batch['input_ids'],
                      'attention_mask': batch['attention_mask']},
                unit_locations={"base" : batch['intervention_locations'].permute(1, 0, 2).tolist()})
            
            ### We assume for now only one-block defence and one-block attacks.
            # TODO adjust indexing here for multi-block defences
            original_input_representations = torch.vstack([output.unsqueeze(0) for output in intervention_outputs[0][1]])     
            
            # TODO iterate over multiple defences 
            # for defence_layer, defence_modules  in defence_config['defences'].items():
            defence_layer, defence_module = list(defence_config['defences'].items())[0]

            with torch.no_grad():
                corruption_module = defence_module['adversarial_intervention_module']
                corrupted_input_reps = corruption_module(original_input_representations)  # these are our "inputs"
                defensive_block = defence_module['next_mlp_block']
                original_output_reps = defensive_block(original_input_representations)  # these are our "labels"

            predicted_outpur_reps = defensive_block.intervene_forward(corrupted_input_reps)
            
            loss = defence_criterion(
                input=predicted_outpur_reps,
                target=original_output_reps)
            
            defence_optimizer.zero_grad()
            loss.backward()
            defence_optimizer.step()

            epoch_loss += loss.detach()

        mean_epoch_loss = (epoch_loss / batch_idx).item()

        mean_epoch_losses += mean_epoch_loss
    
    mean_loss = mean_epoch_losses / defence_config['epochs']

    defence_results = {
        'mean_loss' : mean_loss
    }

    return defence_results  


def get_safety_from_defence_results(defence_results):
    """
    TODO correct this with the real safety 
    """
    # assuming frobenius norm criterion...
    return 1 - defence_results['mean_loss']


def get_performance_from_defence_results(defence_results):
    """
    TODO correct this with the real performance 
    """
    # assuming frobenius norm criterion
    return 1 - defence_results['mean_loss']


def custom_defence(model, tokenizer, defence_config, attack_data_dict, kwargs):

    # intervenable model is used to retrieve the training inputs
    intervenable_model = pyvene.IntervenableModel(defence_config['intervenable_config'], model)
    intervenable_model.disable_model_gradients()
    defence_dataloader = get_defence_dataloader(model, tokenizer, defence_config, attack_data_dict)
    defence_optimizer = get_defence_optimizer(defence_config, kwargs['learning_rate'])
    defence_criterion = get_defence_loss_criterion(defence_config)
    defence_results = defence_training_loop(
        defence_config, 
        defence_dataloader, 
        intervenable_model, 
        defence_criterion, 
        defence_optimizer,
        kwargs)

    defence_config['safety'] = get_safety_from_defence_results(defence_results)
    defence_config['performance'] = get_performance_from_defence_results(defence_results)

    return defence_results


def get_defence_optimizer(defence_config, learning_rate):
    parameters_for_optimizer = []
    for adv_intervention_layer, defence_modules_dict in defence_config['defences'].items():
        defence_block = defence_modules_dict['next_mlp_block']
        parameters_for_optimizer.extend([param for param in defence_block.parameters() if param.requires_grad])
    return torch.optim.Adam(parameters_for_optimizer, lr=learning_rate)


def get_defence_dataloader(model, tokenizer, defence_config, attack_data_dict):
    data_module = get_red_teaming_data_module(
        model, tokenizer, attack_data_dict, defence_config)

    return DataLoader(
            data_module['train_dataset'],
            batch_size=defence_config['batch_size'],
            collate_fn=data_module['data_collator']
        )


def get_toxicity(attack_results):
    """
    TODO implement
    """
    return 1 - attack_results.training_loss


def absorb_defender_adaptor(model, defence_config, kwargs):
    
    for defence_layer, defence_module in defence_config['defences'].items():
        if kwargs['verbose']: print(f'Absorbing defence LoRA in layer {defence_layer+1} mlp gate proj...')
        defensive_block = defence_module['next_mlp_block']

        defensive_lora_adaptor = torch.matmul(
            defensive_block.gate_B.weight, 
            defensive_block.gate_A.weight)

        model.model.layers[defence_layer+1].mlp.gate_proj.weight = torch.nn.Parameter(
            model.model.layers[defence_layer+1].mlp.gate_proj.weight + 
            (defence_config['absortion_scaling'] * defensive_lora_adaptor))

    return model


def evolve_attack_config(model, layer, prev_attack_config, kwargs):
    """
    TODO implement
    """
    attack_config = init_single_layer_attack_config(model, layer, kwargs)
    attack_config['dataset_size'] = prev_attack_config['dataset_size'] + 50 
    if attack_config['dataset_size'] > kwargs['max_red_teaming_dataset_size']:
        attack_config['dataset_size'] = kwargs['max_red_teaming_dataset_size']
    return attack_config


def evolve_defence_config(model, attack_config, attacked_model, prev_defence_config, kwargs):
    """
    TODO implement
    """
    defence_config = init_custom_defence_config(model, attack_config, attacked_model, kwargs)
    defence_config['dataset_size'] = prev_defence_config['dataset_size'] + 50 
    if defence_config['dataset_size'] > kwargs['max_red_teaming_dataset_size']:
        defence_config['dataset_size'] = kwargs['max_red_teaming_dataset_size']
    return defence_config


def pprint_attack_config(attack_config):
    # TODO implement selective reporting...
    pprint(attack_config)
    