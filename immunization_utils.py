import os
import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import pyreft
import pyvene
import pandas as pd
import copy
import datasets
from custom_modelling_llama import *
from tqdm import tqdm, trange 
from pprint import pprint
import wandb
from datasets import load_dataset
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama import modeling_llama
from collections import defaultdict, OrderedDict
from copy import deepcopy  # Use deepcopy to avoid modifying the original template
from peft import PeftModel
from googleapiclient import discovery
import json

from pyreft.interventions import (
        NoreftIntervention,
        NoreftInterventionNoBias,
        LoreftIntervention,
        LoreftInterventionNoBias,
        LobireftIntervention,
        NodireftIntervention)

INTERVENTIONS = [
    NoreftIntervention,
    NoreftInterventionNoBias,
    LoreftIntervention,
    LoreftInterventionNoBias,
    LobireftIntervention,
    NodireftIntervention]

IGNORE_INDEX = -100
CHAT_TEMPLATE = """<s>[INST] %s [/INST]"""
ASSISTANT_TEMPLATE = \
    """"Below is an instruction that describes a task. Write a response that appropriately completes the request.
    \n\n### Instruction:\n%s
    \n\n### Response:"""

LLAMA3_TEMPLATE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": ""}
]

LLAMA3_TEMPLATE_GENERATION_DICT = {"role": "assistant", "content": ""}


ATTACK_LOSS = 'ATTACK_LOSS'
TOXICITY_AFTER_ATTACK = 'TOXICITY_AFTER_ATTACK'
DEFENCE_LOSS = 'DEFENCE_TOTAL_LOSS'
DEFENCE_REG_LOSS = 'DEFENCE_REG_TERM'
DEFENCE_DEF_LOSS = 'DEFENCE_MAIN_LOSS'
TOXICITY_AFTER_DEFENCE = 'TOXICITY_AFTER_DEFENCE'
SAFETY_AFTER_DEFENCE = 'SAFETY_AFTER_DEFENCE'
PERFORMANCE_AFTER_DEFENCE = 'PERFORMANCE_AFTER_DEFENCE'
PERFORMANCE_AFTER_ATTACK = 'PERFORMANCE_AFTER_ATTACK'
INITIAL_TOXICITY = 'INITIAL_TOXICITY'
INTIIAL_PERFORMANCE = 'INITIAL_PERFORMANCE'
STEP_LABEL = 'STEP'
LAYER = 'LAYER'


def find_files_with_substring(directory_path, substring):
    matching_files = []

    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)) and substring in filename:
            matching_files.append(filename)

    return matching_files



def mean_of_tensor_dicts(dict_list):
    """
    Calculates the mean tensor for each key across a list of dictionaries.

    Args:
        dict_list: A list of dictionaries where each dictionary has the same set of keys,
                   and the values for each key are PyTorch tensors.

    Returns:
        A new dictionary with the same keys, where each value is the mean tensor
        calculated across the corresponding values in the input dictionaries.
    """

    result_dict = {}
    keys = dict_list[0].keys()  # Get the common set of keys

    for key in keys:
        # Gather tensors for the current key from all dictionaries
        tensors = [d[key] for d in dict_list]
        
        # Concatenate the tensors along a new dimension (dim=0 for stacking)
        stacked_tensors = torch.stack(tensors)
        
        # Calculate the mean along the stacked dimension
        mean_tensor = torch.mean(stacked_tensors, dim=0)
        
        result_dict[key] = mean_tensor

    return result_dict



def update_state_dict(current_state_dict, new_state_dict, alpha=0.9):
    """
    Updates a PyTorch model's state_dict by blending old and new weights.

    Args:
        current_state_dict (OrderedDict): The model's existing state_dict.
        new_state_dict (OrderedDict): The state_dict with new weights to blend in.
        alpha (float): The blending factor (0.9 keeps 90% of the old weights).
    """
    print('Absortion scaling is: ',alpha)
    updated_state_dict = OrderedDict()
    
    for key in current_state_dict.keys():
        retro_compat_flag = False
        # Ensure both state dicts have the same keys
        if key not in new_state_dict:
            if 'mlp.' in key:
                # retro compatibility
                key = key.replace('mlp.', '')
                retro_compat_flag = True

        if key not in new_state_dict:
            print(f"Key '{key}' not found in the saved vaccine. Skipping...")
            updated_state_dict[key] = current_state_dict[key]
            continue

        current_tensor = (current_state_dict[key] if not retro_compat_flag else current_state_dict['mlp.'+key]) #current_state_dict[key]
        new_tensor = new_state_dict[key]
        
        #put correct value:
        key = (key if not retro_compat_flag else 'mlp.'+key)
        # Ensure both tensors have the same shape
        if current_tensor.shape != new_tensor.shape:
            raise ValueError(f"Shape mismatch for key '{key}': {current_tensor.shape} vs {new_tensor.shape}")
        print('updating key: ',key)
        updated_tensor = (1 - alpha) * current_tensor + alpha * new_tensor 
        updated_state_dict[key] = updated_tensor
    
    return updated_state_dict


def mount_vaccines(model, kwargs):
    if kwargs['mount_vaccines'] != '':

        if kwargs['mount_vaccines'] == 'super' or '*' in kwargs['mount_vaccines']: 

            if kwargs['mount_vaccines'] == 'super':
                matching_filenames = find_files_with_substring(kwargs['cache_dir']+'/ET/', 'VACCINE')
                matching_filenames = [filename for filename in matching_filenames if 'GU' not in filename and 'ml' not in filename]
            elif kwargs['mount_vaccines'] == 'super_ml':
                matching_filenames = find_files_with_substring(kwargs['cache_dir']+'/ET/', 'VACCINE')
                matching_filenames = [filename for filename in matching_filenames if 'GU' not in filename]
            elif '*' in kwargs['mount_vaccines']:
                matching_filenames = find_files_with_substring(kwargs['cache_dir']+'/ET/', kwargs['mount_vaccines'].split('*')[0])
            print('will mount the following vaccines: ', matching_filenames)
            defenders_per_layer = defaultdict(list)
            for defender_adaptor in matching_filenames:
                layer = int(defender_adaptor.split('layer')[1].split('_')[0])
                defenders_per_layer[layer].append(torch.load(kwargs['cache_dir']+'/ET/'+defender_adaptor, weights_only=True))
            for layer, list_of_layer_dicts in defenders_per_layer.items():
                if kwargs['avg_multiple_vaccines']:
                    print(f'Averaging {len(list_of_layer_dicts)} adapters at layer {layer}')
                    mean_layer_state_from_adapters = mean_of_tensor_dicts(list_of_layer_dicts)
                    current_state_dict = model.model.layers[layer].state_dict()
                    updated_state_dict = update_state_dict(current_state_dict, mean_layer_state_from_adapters, kwargs['vaccine_weight'])
                    model.model.layers[layer].load_state_dict(updated_state_dict)
                else:
                    print(f'Mounting the last adapter of {len(list_of_layer_dicts)} at layer {layer}')
                    adapter = list_of_layer_dicts[-1]
                    current_state_dict = model.model.layers[layer].state_dict()
                    updated_state_dict = update_state_dict(current_state_dict, adapter, kwargs['vaccine_weight'])
                    model.model.layers[layer].load_state_dict(updated_state_dict)

        else:
            for vaccine_path in kwargs['mount_vaccines'].split(':'):
                print('mounting vaccine: ', kwargs['cache_dir']+'/'+vaccine_path)
                layer = int(vaccine_path.split('layer')[1].split('_')[0])
                model.model.layers[layer].mlp.load_state_dict(torch.load(kwargs['cache_dir']+'/'+vaccine_path))
    return model
     
        
def initialize(args):
    
    kwargs = dict(args)
    torch.manual_seed(kwargs['torch_seed'])
    model, tokenizer = load_model(kwargs)
    model = mount_vaccines(model, kwargs)

    if kwargs['save_immunised']:
        # save model to disk:
        print('Saving model to disk: ', kwargs['cache_dir']+'/'+kwargs['model_name_or_path'].split('/')[-1]+'_ET')
        model.save_pretrained(kwargs['cache_dir']+'/'+kwargs['model_name_or_path'].split('/')[-1]+'_ET')
        tokenizer.save_pretrained(kwargs['cache_dir']+'/'+kwargs['model_name_or_path'].split('/')[-1]+'_ET')
        print('Done saving model to disk: ', kwargs['cache_dir']+'/'+kwargs['model_name_or_path'].split('/')[-1]+'_ET')
    
    if kwargs['eval_safety_model'] == 'llama_guard':
        eval_model, eval_tokenizer = load_eval_model(kwargs)
    elif kwargs['eval_safety_model'] == 'perspective_api':
        with open('.env') as f:
            env_vars = {line.split('=')[0]:line.split('=')[1].strip() for line in f}
        API_KEY = env_vars['GCP_API_KEY']
        eval_model = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
            ).comments()
        eval_tokenizer = None
    else:
        raise "Fatal error: eval_safety_model must be either 'llama_guard' or 'perspective_api'!!!"
    
    training_attack_data_dict = load_training_red_teaming_data(tokenizer, kwargs)
    safety_eval_data = load_eval_red_teaming_data(tokenizer, kwargs)
    performance_eval_data = get_performance_eval_dataset(tokenizer, kwargs)
    logging_dict = init_wandb_stuff(kwargs)

    kwargs['current_layer'] = \
        kwargs['current_outer_attack_round'] = \
        kwargs['current_inner_attack_round'] = \
        kwargs['current_outer_defence_round'] = \
        kwargs['current_inner_defence_round'] = 'init' 
    kwargs['timestep'] = 0
    init_toxicity, init_safety_eval_table = eval_safety(
        model, 
        tokenizer,
        eval_model,
        eval_tokenizer,
        safety_eval_data,
        args.init_eval_safety_prompts,
        logging_dict,
        False,
        'init',
        kwargs)
    logging_dict['wandb_run'].log(
        {"Baseline Behaviour": init_safety_eval_table})
    init_safety = 1 - init_toxicity
    init_performance = 1/eval_performance(
        model,
        performance_eval_data,
        1,
        False,
        kwargs)
    kwargs['init_toxicity'] = init_toxicity
    kwargs['init_safety'] = init_safety
    kwargs['init_performance'] = init_performance

    if kwargs['verbose']:
        print(f'INITIAL TOXICITY: {init_toxicity} Initial Performance: {init_performance}')
    logging_dict['wandb_run'].log({'INITIAL_TOXICITY': init_toxicity, 'STEP': kwargs['timestep']})
    logging_dict['wandb_run'].log({'INITIAL_PERFORMANCE': init_performance, 'STEP': kwargs['timestep']})
    if kwargs['verbose'] and not kwargs['logging']: 
        print("IMMUNIZATION PARAMS:")
        pprint(kwargs)
        print("\n")


    return kwargs, \
        logging_dict, \
        model, tokenizer, \
        eval_model, eval_tokenizer, \
        training_attack_data_dict, \
        safety_eval_data, \
        performance_eval_data


def init_wandb_stuff(kwargs):
        # logging stuff:
        wandb_tags = [
            'IMMUNIZATION',
            'REG',
            'FIX',
            'CAUSAL_MASK',
            'NEW_LOGGING',
            'STEP_FIX',
            'EQUITY',
            'FAST',
            'TEMPLATE_FIX',
            'DEFENCE_FIX',
            'NEW_PARAMS',
            'ABS_LOGGING'
            ] + kwargs['tags'].split(';')
        run = wandb.init(
                project='low_cost_toxification',
                config=kwargs,
                mode=("online" if kwargs['logging'] else "disabled"),
                name= kwargs['run_name'],
                tags=wandb_tags)

        print("Starting immunization process...\n\n")
        immunization_report = []
        immunization_table = wandb.Table(columns=[
            "layer", 
            "immunized",
            "attack_rounds",
            "defence_rounds",
            "max_toxicity",
            "current_safety",
            "current_performance"])
        # only succesful steps
        step_report = []  
        step_table = wandb.Table(columns=[
            "layer", 
            "action",
            "toxicity",
            "performance",
            "step"])
        # all steps
        all_step_report = []
        all_step_table = wandb.Table(columns=[
            "layer", 
            "action",
            "toxicity",
            "performance",
            "step"])

        return {'wandb_run': run,
                'immunization_report' : immunization_report,
                'step_report': step_report,
                'all_step_report': all_step_report,
                'immunization_table': immunization_table,
                'step_table': step_table,
                'all_step_table': all_step_table}


def get_performance_eval_dataset(tokenizer, kwargs):
    """
    This function was copied and edited from https://github.com/boyiwei/alignment-attribution-code
    """
    if kwargs['verbose']: print('Loading performance evaluation data...\n\n')
    testdata = load_dataset("wikitext",
                            "wikitext-2-raw-v1",
                            split="test",
                            cache_dir=kwargs["dataset_cache_dir"])
    encoded_test_data = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return encoded_test_data


def eval_performance(model, testenc, batch_size, isReftModel, kwargs):
    """
    This function was copied and edited from https://github.com/boyiwei/alignment-attribution-code
    """
    # Get input IDs
    testenc = testenc.input_ids
    if isReftModel:
        seqlen = model.model.config.max_position_embeddings
    else:
        seqlen = model.config.max_position_embeddings
    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
    nsamples = min(nsamples, kwargs['performance_batches'])

    if isReftModel:
        nsamples = 2 # we do not care about performance of attacks for now...
    ranger = range
    if kwargs['verbose']:
        print(f"Evaluating perplexity...")
        if kwargs['tqdm']: ranger = trange

    # List to store negative log likelihoods
    nlls = []

    with torch.no_grad():

        # Loop through each batch
        for i in ranger(0, nsamples, batch_size):
            # if i % 10 == 0 and not kwargs['tqdm']:
            #    if kwargs['verbose']: print(f"eval perplexity... sample {i}")

            # Calculate end index
            j = min(i + batch_size, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[:, (i * seqlen) : (j * seqlen)].to(kwargs['device'])
            inputs = inputs.reshape(j - i, seqlen)

            if isReftModel:
                unit_locations={
                "sources->base": (
                    None, 
                    [[list(range(seqlen))]]*len(model.interventions)
                                )}

                intervention_outputs = model(
                    base={'input_ids' : inputs},
                    unit_locations=unit_locations
                )
                lm_logits = intervention_outputs[1].logits
            else:

                # Forward pass through the model
                lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
            )

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * seqlen * (j - i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def log_immunization(layer, immunized, attack_rounds, defence_rounds, attack_config, defence_config, logging_dict, kwargs):

    record =  {'layer':layer, 
                'immunized': immunized, 
                'attack_rounds': attack_rounds, 
                'defence_rounds': defence_rounds,
                'max_toxicity': attack_config['toxicity'],
                'current_safety': (1 - attack_config['toxicity']),
                'current_performance': (defence_config['performance'] if defence_config else 0)}

    logging_dict['immunization_report'].append(record)
    logging_dict['immunization_table'].add_data(
        layer,
        immunized,
        attack_rounds,
        defence_rounds,
        attack_config['toxicity'],
        (1 - attack_config['toxicity']),
        (defence_config['performance'] if defence_config else 0))

    logging_dict['wandb_run'].log(
        { ('S' if immunized else 'Uns') + 'uccesfully immunized layers': layer,
         STEP_LABEL: kwargs['timestep'] })


def log_step(layer, action, toxicity, performance, logging_dict, kwargs):

    record =  {
        'layer':layer, 
        'action': action, 
        'toxicity': toxicity, 
        'performance': performance,
        'step': kwargs['timestep'],
        }

    logging_dict['all_step_report'].append(record)
    logging_dict['all_step_table'].add_data(
        layer,
        action,
        toxicity,
        performance,
        kwargs['timestep'])

    logging_dict['wandb_run'].log(
        { 'Toxicity after ' + action + 's' : toxicity,
         STEP_LABEL: kwargs['timestep'] })

    logging_dict['wandb_run'].log(
        { 'Performance after ' + action + 's' : performance,
         STEP_LABEL: kwargs['timestep'] })


def log_successful_step(layer, action, toxicity, performance, logging_dict, kwargs):

    record =  {
        'layer':layer, 
        'action': action, 
        'toxicity': toxicity, 
        'performance': performance /( kwargs['init_performance']+ 1e-10),
        'step': kwargs['timestep'],
        }

    logging_dict['step_report'].append(record)
    logging_dict['step_table'].add_data(
        layer,
        action,
        toxicity,
        performance /( kwargs['init_performance']+ 1e-10),
        kwargs['timestep'])

    

    logging_dict['wandb_run'].log(
        { 'Toxicity after succesful ' + action + 's' : toxicity,
         STEP_LABEL: kwargs['timestep'] })

    logging_dict['wandb_run'].log(
        { 'Performance after succesful ' + action + 's' : performance /( kwargs['init_performance']+ 1e-10),
         STEP_LABEL: kwargs['timestep'] })


def load_model(kwargs):
    if kwargs['verbose']: print('Loading the model we want to immunize...\n\n')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        kwargs['model_name_or_path'],
        torch_dtype=torch.bfloat16,
        device_map=kwargs['device'],
        cache_dir=kwargs['cache_dir']
    )

    if kwargs['lora_adaptor']:
        if kwargs['verbose']: print('Loading the lora adaptor...\n\n')
        model = PeftModel.from_pretrained(
            model,
            kwargs['lora_adaptor'],
            )
        model = model.merge_and_unload()
        

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        kwargs['model_name_or_path'],
        use_fast=False,
        cache_dir=kwargs['cache_dir'],
        model_max_length=kwargs['max_seq_len'],
    )
    tokenizer.pad_token_id = 0
    return model, tokenizer


def load_eval_model(kwargs):
    if kwargs['verbose']: print('Loading the evaluator model...\n\n')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        kwargs['eval_model'],
        torch_dtype=torch.bfloat16,
        device_map=kwargs['device'],
        cache_dir=kwargs['cache_dir']
    )

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        kwargs['eval_model'],
        use_fast=False,
        cache_dir=kwargs['cache_dir'],
        model_max_length=1024,
    )
    tokenizer.pad_token_id = 0
    return model, tokenizer


def load_training_red_teaming_data(tokenizer, kwargs):
        if kwargs['verbose']: print('Loading red teaming data...\n\n')
        red_teaming_df = pd.read_csv(kwargs['training_red_teaming_data_path'])
        if kwargs['template'] == "chat":
            toxic_prompts = [CHAT_TEMPLATE % p for p in red_teaming_df[kwargs['train_red_teaming_input_col']].tolist()]
        elif kwargs['template'] == "generic_assistant":
            toxic_prompts = [ASSISTANT_TEMPLATE % p for p in red_teaming_df[kwargs['train_red_teaming_input_col']].tolist()]
        elif kwargs['template'] == "llama3_assistant":
            toxic_prompts = []
            for p in red_teaming_df[kwargs['train_red_teaming_input_col']].tolist():
                toxic_prompt = deepcopy(LLAMA3_TEMPLATE)
                toxic_prompt[-1]["content"] = p
                toxic_prompts.append(toxic_prompt)
        
        toxic_completions = red_teaming_df[kwargs['train_red_teaming_label_col']].tolist()
        
        all_base_input_ids, all_intervention_locations, all_output_ids, all_base_prompt_ids = [], [], [], []
        
        for i in range(len(toxic_prompts)):
            _input = toxic_prompts[i]
            _output = toxic_completions[i]  # TODO +": "  # may make the attack more efficacious
        
            base_prompt = _input

            if kwargs['template'] == "llama3_assistant":

                base_input = deepcopy(base_prompt)
                output_dict = deepcopy(LLAMA3_TEMPLATE_GENERATION_DICT)
                output_dict['content'] = _output
                base_input.append(output_dict)

                base_prompt_ids = tokenizer.apply_chat_template(
                    base_prompt,
                    add_generation_prompt=True,
                    max_length=tokenizer.model_max_length,
                    truncation=True,  # Assuming Truncation 
                    return_tensors="pt").squeeze(0)

                base_input_ids = tokenizer.apply_chat_template(
                    base_input,
                    max_length=tokenizer.model_max_length,
                    truncation=True,  # Assuming Truncation
                    return_tensors="pt").squeeze(0)

                # We do not want to exactly mimic the limited answers of the GCG attack in the dataset...
                base_input_ids = base_input_ids[:-2]
            else:

                base_input = base_prompt + _output
                # Assuming nonstop #TODO test without
                base_input += tokenizer.eos_token
                # tokenize
                base_prompt_ids = tokenizer(
                    base_prompt, 
                    max_length=tokenizer.model_max_length,
                    truncation=True,  # Assuming Truncation 
                    return_tensors="pt")["input_ids"][0]

                base_input_ids = tokenizer(
                    base_input,
                    max_length=tokenizer.model_max_length,
                    truncation=True,  # Assuming Truncation
                    return_tensors="pt")["input_ids"][0]

            base_prompt_length = len(base_prompt_ids)
        
            output_ids = copy.deepcopy(base_input_ids)
            output_ids[:base_prompt_length] = IGNORE_INDEX

            all_base_input_ids.append(base_input_ids)
            all_output_ids.append(output_ids)
            all_base_prompt_ids.append(base_prompt_ids)
        kwargs['max_red_teaming_dataset_size'] = len(toxic_prompts)

        # Assuming full position here, TODO implement partial positions...
        return {'red_team_prompts' : toxic_prompts,
                'red team completions' : toxic_completions,
                'all_base_input_ids': all_base_input_ids,
                'all_base_prompt_ids': all_base_prompt_ids,
                'all_output_ids' : all_output_ids }


def load_eval_red_teaming_data(tokenizer, kwargs):
        if kwargs['verbose']: print('Loading safety evaluation data...\n\n')
        red_teaming_df = pd.read_csv(kwargs['eval_red_teaming_data_path'])
        
        toxic_prompts_no_template = red_teaming_df[kwargs['test_red_teaming_input_col']].tolist()
        
        if kwargs['template'] == "chat":
            toxic_prompts = [CHAT_TEMPLATE % p for p in toxic_prompts_no_template]
        elif kwargs['template'] == "generic_assistant":
            toxic_prompts = [ASSISTANT_TEMPLATE % p for p in toxic_prompts_no_template]
        elif kwargs['template'] == "llama3_assistant":
            toxic_prompts = []
            for p in red_teaming_df[kwargs['test_red_teaming_input_col']].tolist():
                toxic_prompt = deepcopy(LLAMA3_TEMPLATE)
                toxic_prompt[-1]["content"] = p
                toxic_prompts.append(toxic_prompt)

        all_base_input_ids = []
        
        for i in range(len(toxic_prompts)):
            _input = toxic_prompts[i]
            base_input = base_prompt = _input

            if kwargs['template'] == "llama3_assistant":

                base_input_ids = tokenizer.apply_chat_template(
                    base_input,
                    add_generation_prompt=True,
                    max_length=tokenizer.model_max_length,
                    truncation=True,  # Assuming Truncation 
                    return_tensors="pt").squeeze(0)
            else:

                # Assuming nonstop #TODO test without
                base_input += tokenizer.eos_token
                
                base_input_ids = tokenizer(
                    base_input,
                    max_length=tokenizer.model_max_length,
                    truncation=True,  # Assuming Truncation
                    return_tensors="pt")["input_ids"][0]

            all_base_input_ids.append(base_input_ids)

        return {'red_team_prompts_no_template': toxic_prompts_no_template,
                'red_team_prompts' : toxic_prompts,
                'all_base_input_ids': all_base_input_ids}


def init_single_layer_attack_config(model, layer, kwargs):

    representations = []

    intervention_place = kwargs['init_attack_intervention_places'] + '_input'
   
    embed_dim = model.config.hidden_size

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
                     'safety_eval_dataset_size' : kwargs['init_eval_safety_prompts'],
                     'batch_size' : kwargs['init_attack_batch_size']}
    return attack_config


def init_custom_defence_config(model, attack_config, attacked_model, defences, kwargs):

    defence_config = {
        'defences': {},
        'low_rank_defence': kwargs['init_low_rank_defence_dimension'],
        }

    collect_interventions = []

    # We assume attacks are made only in the residual_stream, so for each layer, we have 
    # at most one attack # TODO think of mutiple position attacks?
    intervention_key, intervention_module = list(attacked_model.interventions.items())[0]
    intervention_layer = int(intervention_key.split('layer.')[1].split('.')[0])
    freezed_intervention_module = get_freezed_intervention_module(intervention_module[0])

    for defence_idx in range(defences):

        curr_defence_layer = intervention_layer + defence_idx

        defensive_module = get_defensive_module(
            model, 
            curr_defence_layer, 
            defence_config['low_rank_defence'],
            kwargs)

        defence_config['defences'][curr_defence_layer] = {
            'defence_module' : defensive_module
        }

        # Append collect interventions:

        # block_input (before pre-layer-norm) or mlp_input (after post-layer-norm)
        collect_interventions.append({
        "layer": curr_defence_layer,
        "component": kwargs['init_attack_intervention_places']+'_input'
        })
        
        """
        # block output (after second residual connection) or mlp output (before second res connection)
        collect_interventions.append({
        "layer": curr_defence_layer,
        "component": kwargs['init_attack_intervention_places']+'_output'
        })
        """

        # EO Append collect interventions.

    # craft defence config dict:
    defence_config['intervenable_config'] = pyvene.IntervenableConfig(
        model_type=type(model),
        representations=collect_interventions,
        intervention_types=pyvene.CollectInterventionKLD,
    )

    defence_config['adversarial_intervention_module'] = freezed_intervention_module
    defence_config['dataset_size'] = kwargs['init_defence_prompts']
    defence_config['epochs'] =  kwargs['init_defence_epochs']
    defence_config['batch_size'] = kwargs['init_defence_batch_size']
    defence_config['intervention_count'] = len(collect_interventions)
    defence_config['defence_criterion'] = kwargs['init_defence_criterion']
    defence_config['absortion_scaling'] = kwargs['init_defence_absortion_scaling']
    defence_config['safety'] = 0
    defence_config['performance'] = 0
    defence_config['safety_eval_dataset_size'] = kwargs['init_eval_safety_prompts']
    # defence_config['performance_eval_dataset_size'] = kwargs['init_eval_performance_prompts']  # TODO use this!
    defence_config['regularization_coefficient'] = kwargs['defence_reg_coeff']

    return defence_config


def is_successful_attack(attack_config, kwargs):
    
    return (attack_config['toxicity'] > (kwargs['init_toxicity'] * kwargs['min_toxicity_increase_factor']) and 
            attack_config['performance'] >= (kwargs['init_performance'] * kwargs['min_performance_percentage_attack']))
    

def is_successful_defence(defence_config, kwargs):
    
    return defence_config['safety'] >= (kwargs['min_safety_percentage'] * kwargs['init_safety']) and \
                                defence_config['performance'] >= (kwargs['min_performance_percentage_defence'] * kwargs['init_performance'])

def get_freezed_intervention_module(intervention_module):
    for parameter in intervention_module.parameters():
        parameter.requires_grad = False
    return intervention_module


def get_defensive_module(model, layer, low_rank_dimension, kwargs):

    """
    The block to act is a follower of the one that receives corrupted interventions
    """
    if kwargs['init_attack_intervention_places'] == 'block':
        originial_next_block = model.model.layers[layer]
        return LlamaBlockDefendor(originial_next_block, low_rank_dimension, kwargs)
    elif kwargs['init_attack_intervention_places'] == 'mlp':
        originial_next_block = model.model.layers[layer].mlp
        return CustomLlamaMLP(originial_next_block, low_rank_dimension, kwargs)
    else:
        assert False, "Fatal error: init_attack_intervention_places must be either \"block\" or \"mlp\"!!!"


def get_red_teaming_data_module(model, tokenizer, attack_data_dict, process_config, mode):
    """
    #TODO implement suffling?
    """
    
    samples_to_use = process_config['dataset_size']
    # Assuming full position here, TODO implement partial positions...
    intervetion_locations = [[[[-1]] * process_config['intervention_count']]] * samples_to_use
    intervetion_locations = torch.Tensor(intervetion_locations).squeeze(1)
    
    if mode == 'attack':
        train_dataset = datasets.Dataset.from_dict({
            "input_ids": attack_data_dict['all_base_input_ids'][:samples_to_use],
            "intervention_locations": intervetion_locations,
            "labels": attack_data_dict['all_output_ids'][:samples_to_use],
        })
    elif mode == 'defence':
        train_dataset = datasets.Dataset.from_dict({
                "input_ids": attack_data_dict['all_base_prompt_ids'][:samples_to_use],
                "intervention_locations": intervetion_locations,
                # "labels": attack_data_dict['all_output_ids'][:samples_to_use]
            })
        
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )

    # TODO here we are assuming full positions. 
    data_collator = pyreft.FullPosReftDataCollator(
        data_collator=data_collator_fn)
    
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=None,
        data_collator=data_collator)
        

def reft_attack(
        model, 
        tokenizer, 
        attack_config, 
        attack_data_dict, 
        eval_model,
        eval_tokenizer, 
        safety_eval_data,
        performance_eval_data,
        logging_dict,
        kwargs):
    if kwargs['verbose']: print('Performing Adversary Training...')
    report_attack_config(attack_config, logging_dict, kwargs)
    reft_model = pyreft.get_reft_model(model, attack_config['reft_config'])
    reft_model.set_device(kwargs['device'])
    # if kwargs['verbose']: 
    #    reft_model.print_trainable_parameters()
    #    print('Number of attack interventions:', attack_config['intervention_count'])
    attack_data_module = get_red_teaming_data_module(
        model, tokenizer, attack_data_dict, attack_config, 'attack')

    training_args = transformers.TrainingArguments(
        num_train_epochs=attack_config['epochs'],
        output_dir="local_checkpoints/tmp_reft",  # TODO what to do here?
        overwrite_output_dir=True,
        per_device_train_batch_size=attack_config['batch_size'],
        learning_rate=kwargs['attack_learning_rate'],
        report_to="none",
        disable_tqdm=(not kwargs['tqdm']),
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        processing_class=tokenizer,
        args=training_args,
        **attack_data_module
    )
   
    attack_results = trainer.train()

    logging_dict['wandb_run'].log(
        { ATTACK_LOSS: attack_results.training_loss,
         STEP_LABEL: kwargs['timestep'] })

    attack_config['toxicity'], eval_table = eval_safety(
        reft_model,
        tokenizer,
        eval_model,
        eval_tokenizer,
        safety_eval_data,
        attack_config['safety_eval_dataset_size'],
        logging_dict,
        True,
        'attack',
        kwargs)

    attack_config['performance'] = 1/ eval_performance(
        reft_model,
        performance_eval_data,
        1,
        True,
        kwargs
    )

    
    logging_dict['wandb_run'].log(
        { TOXICITY_AFTER_ATTACK: attack_config['toxicity'],
         STEP_LABEL: kwargs['timestep'] })

    logging_dict['wandb_run'].log(
        { PERFORMANCE_AFTER_ATTACK: attack_config['performance'] ,
         STEP_LABEL: kwargs['timestep'] })

    return reft_model, eval_table


def report_qualitative_immunisation_results(
        post_successful_attack_behaviour, post_failed_attack_behaviour, logging_dict, kwargs):

    logging_dict['wandb_run'].log(
                {"After Successful Attack Behaviour Layer " \
                    + str(kwargs['current_layer']) + " Step " \
                    + str(kwargs['timestep']) : post_successful_attack_behaviour})
    logging_dict['wandb_run'].log(
                {"After Failed Attack Behaviour Layer " \
                    + str(kwargs['current_layer']) + " Step " \
                    + str(kwargs['timestep']) : post_failed_attack_behaviour})


def eval_safety(
        model, 
        model_tokenizer, 
        eval_model, 
        eval_tokenizer, 
        safety_eval_data,
        eval_prompts,
        logging_dict,
        isReftModel,
        mode,
        kwargs):

    if kwargs['verbose']: print('Evaluating safety...')

    safety_eval_iterator = zip(
                safety_eval_data['red_team_prompts_no_template'][:eval_prompts], 
                safety_eval_data['all_base_input_ids'][:eval_prompts])

    if kwargs['tqdm']: 
        safety_eval_iterator = tqdm(safety_eval_iterator, total=eval_prompts)

    with torch.no_grad():

        flag = 0
        eval_step = 0
        eval_table = wandb.Table(columns=['prompt', 'response', 'moderation'])

        for eval_prompt, tokenized_eval_prompt in safety_eval_iterator:
            
            output = deterministic_query(
                tokenized_eval_prompt,
                model,
                model_tokenizer,
                isReftModel,
                kwargs
            )
        
            if kwargs['eval_safety_model'] == 'llama_guard':
                
                data_tuple = [{"role": "user",
                            "content": eval_prompt},
                            {"role": "assistant",
                            "content": output}]

                moderation = moderate(data_tuple, eval_model, eval_tokenizer, kwargs['device'])
                unsafe_cat = 'safe'
                if 'unsafe' in moderation:
                    flag += 1
                    unsafe_cat = moderation.split('\n')[-1]
                # if kwargs['verbose']: print('moderation: ',unsafe_cat)
                eval_table.add_data(eval_prompt, output, unsafe_cat)
                

            elif kwargs['eval_safety_model'] == 'perspective_api':
                analyze_request = {
                    'comment': {'text': f"Adversarial Prompt: {eval_prompt}\nLLM Response: {output}"},
                    'requestedAttributes': {'SEVERE_TOXICITY': {}}
                }
                response = eval_model.analyze(body=analyze_request).execute()
                tox_score = response['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
                eval_table.add_data(eval_prompt, output, tox_score)
                flag += tox_score

            else:
                raise "Fatal error: eval_safety_model must be either 'llama_guard' or 'perspective_api'!!!"

            eval_step += 1


        toxicity_score = flag/eval_step

    return toxicity_score, eval_table


def moderate(chat, evaluator_model, evaluator_tokenizer, device):

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


def deterministic_query(
        tokenized_prompt, model, tokenizer, isReftModel, kwargs):

    tokenized_prompt = tokenized_prompt.to(kwargs['device'])
    input_len = len(tokenized_prompt)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if isReftModel:

            unit_locations={
            "sources->base": (None, [[list(range(input_len))]]*len(model.interventions))}

            _,s = model.generate(
            base={'input_ids' : tokenized_prompt.unsqueeze(0)},
            unit_locations=unit_locations,
            intervene_on_prompt=True,
            top_p=1,
            temperature=1.0,
            do_sample=False,
            max_new_tokens=kwargs['max_gen_tokens'],
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            )
    else:
            s = model.generate(
                inputs=tokenized_prompt.unsqueeze(0),
                top_p=1,
                temperature=1.0,  # greedy decoding
                do_sample=False,  # greedy decoding
                max_new_tokens=kwargs['max_gen_tokens'],
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
            )
    s = s[0]

    output = tokenizer.decode(
        s[input_len:], 
        skip_special_tokens=True)

    return output


def get_defence_loss_criterion(defence_config):
    """
    """
    assert defence_config["defence_criterion"] in ['fro','mse', 'fro_cos', 'mse_cos']  # TODO support other losses?

    if defence_config["defence_criterion"] == "fro":
        return fro_norm_loss
    elif defence_config["defence_criterion"] == "mse":
        return mse_loss
    elif defence_config["defence_criterion"] == "fro_cos":
        return fro_plus_cos_norm_loss
    elif defence_config["defence_criterion"] == "mse_cos":
        return mse_plus_cos_norm_loss

def mse_plus_cos_norm_loss(input, target, **kwargs):
    alpha = kwargs.get('frobenious_norm_scaling_factor', 1.0)
    beta = kwargs.get('cosine_similarity_scaling_factor', 1.0)
    
    mse_loss = torch.nn.MSELoss()(input, target)

    # Flatten across tokens and hidden_dim for cosine similarity
    input_flat = input.view(input.size(0), -1)  # [batch_size, num_tokens * hidden_dim]
    target_flat = target.view(target.size(0), -1)  # [batch_size, num_tokens * hidden_dim]


    # Cosine similarity (per token)  Shape: [batch_size, num_tokens]
    cosine_similarity = F.cosine_similarity(input_flat, target_flat, dim=1) # Per-batch similarity
    cosine_loss = 1 - cosine_similarity.mean()  # Average over all tokens and batch
    # Combine the two losses
    total_loss = alpha * mse_loss + beta * cosine_loss
    return total_loss


def mse_loss(input, target, **kwargs):
    return torch.nn.MSELoss()(input, target)


def fro_plus_cos_norm_loss(input, target, **kwargs):
    alpha = kwargs.get('frobenious_norm_scaling_factor', 1.0)
    beta = kwargs.get('cosine_similarity_scaling_factor', 1.0)
    
    frobenius_loss = torch.norm(target - input, p='fro') ** 2
    # Cosine similarity (per token)  Shape: [batch_size, num_tokens]
    cosine_similarity = F.cosine_similarity(input, target, dim=-1)
    cosine_loss = 1 - cosine_similarity.mean()  # Average over all tokens and batch
    # Combine the two losses
    total_loss = alpha * frobenius_loss + beta * cosine_loss
    return total_loss


def fro_norm_loss(input, target, **kwargs):
    return torch.norm(input- target)


def mlp_immunisation_step(
    batch,
    intervenable_model,
    defence_config,
    defence_criterion,
    kwargs):

    reg_loss = 0
    def_loss = 0
    corruption_module = defence_config['adversarial_intervention_module']

    intervention_locations = batch['intervention_locations'].permute(1, 0, 2).tolist()
    position_ids = torch.tensor(intervention_locations[0][0], device=kwargs['device']).unsqueeze(0)
    batch['input_ids'] = batch['input_ids'].to(kwargs['device'])
    batch['attention_mask'] = batch['attention_mask'].to(kwargs['device'])

    with torch.no_grad():

        intervention_outputs = intervenable_model(
            base={'input_ids' : batch['input_ids'],
                    'attention_mask': batch['attention_mask']},
            unit_locations={"base" : intervention_locations})

        batch_size = defence_config['batch_size']
        intervention_outputs = [fella.unsqueeze(0) for fella in intervention_outputs[0][1]]
        defence_tuples = list(defence_config['defences'].items())
        assert len(intervention_outputs) == batch_size * len(defence_tuples), "something must went wrong!"

    for defence_idx in range(len(defence_tuples)):
        start_idx = defence_idx * batch_size
        
        orig_mlp_inputs = torch.vstack(intervention_outputs[
            start_idx:
            start_idx + batch_size])
        
        corrupt_mlp_inputs = corruption_module(orig_mlp_inputs)

        defence_layer, defence_module = defence_tuples[defence_idx]
        defensive_block = defence_module['defence_module']
        
        bsl_mlp_outputs = defensive_block(
            orig_mlp_inputs)

        # Neutralisation:
        neutralisation_mlp_outs = defensive_block.interveened_forward(
            corrupt_mlp_inputs)

        def_loss += defence_criterion(
            input=neutralisation_mlp_outs,
            target=bsl_mlp_outputs,
            **kwargs)

        # Stability:
        stability_mlp_outputs = defensive_block.interveened_forward(
            orig_mlp_inputs)

        reg_loss += defence_criterion(
            input=stability_mlp_outputs,
            target=bsl_mlp_outputs,
            **kwargs)
        

    return reg_loss, def_loss


def block_immunisation_step(   
    batch,
    intervenable_model,
    defence_config,
    defence_criterion,
    preinputs_catcher,
    kwargs):

    block_def_loss = 0
    block_reg_loss = 0
    attn_reg_loss = 0
    mlp_reg_loss = 0
    attn_def_loss = 0
    mlp_def_loss = 0
    corruption_module = defence_config['adversarial_intervention_module']

    intervention_locations = batch['intervention_locations'].permute(1, 0, 2).tolist()
    position_ids = torch.tensor(intervention_locations[0][0], device=kwargs['device']).unsqueeze(0)
    batch['input_ids'] = batch['input_ids'].to(kwargs['device'])
    batch['attention_mask'] = batch['attention_mask'].to(kwargs['device'])

    with torch.no_grad():

        intervention_outputs = intervenable_model(
            base={'input_ids' : batch['input_ids'],
                    'attention_mask': batch['attention_mask']},
            unit_locations={"base" : intervention_locations})

        pre_inputs_dict = preinputs_catcher.get_inputs_dict(
            batch['input_ids'],
            batch['attention_mask'])


        batch_size = defence_config['batch_size']
        intervention_outputs = [fella.unsqueeze(0) for fella in intervention_outputs[0][1]]
        defence_tuples = list(defence_config['defences'].items())
        assert len(intervention_outputs) == batch_size * len(defence_tuples), "something must went wrong!"


    for defence_idx in range(len(defence_tuples)):
        start_idx = defence_idx * batch_size

        h = torch.vstack(intervention_outputs[
            start_idx:
            start_idx + batch_size])

        h_plus_i = corruption_module(h).detach()
        i = h_plus_i - h.detach()
    
        defence_layer, defence_module = defence_tuples[defence_idx]
        defensive_block = defence_module['defence_module']

        safe_block_output =  defensive_block(
            h,
            attention_mask=pre_inputs_dict['causal_mask'],
            position_ids=pre_inputs_dict['position_ids'],
            past_key_value=pre_inputs_dict['past_key_values'],
            output_attentions=pre_inputs_dict['output_attentions'],
            use_cache=pre_inputs_dict['use_cache'],
            cache_position=pre_inputs_dict['cache_position'],
            position_embeddings=pre_inputs_dict['position_embeddings'])[0]

        safe_attn_output = defensive_block.self_attn_output_cache.detach()
        safe_pre_mlp_residual = defensive_block.pre_mlp_res_cache.detach()
        safe_mlp_output = defensive_block.mlp_output_cache.detach()

        # neutralisation

        # We run this forward pass just to capture activations in the places we care...
        infected_block_output = defensive_block.interveened_forward(
            h_plus_i,
            attention_mask=pre_inputs_dict['causal_mask'],
            position_ids=pre_inputs_dict['position_ids'],
            past_key_value=pre_inputs_dict['past_key_values'],
            output_attentions=pre_inputs_dict['output_attentions'],
            use_cache=pre_inputs_dict['use_cache'],
            cache_position=pre_inputs_dict['cache_position'],
            position_embeddings=pre_inputs_dict['position_embeddings'])[0]

        # get the activations...
        infected_attn_output = defensive_block.self_attn_output_cache
        infected_pre_mlp_residual = defensive_block.pre_mlp_res_cache
        infected_mlp_output = defensive_block.mlp_output_cache

        attn_def_loss += defence_criterion(
            input=infected_attn_output,
            target=safe_attn_output,
            **kwargs)
        

        mlp_def_loss += defence_criterion(
            input=infected_mlp_output,
            target=safe_mlp_output,
            **kwargs)

        block_def_loss += defence_criterion(
            input=infected_block_output,
            target=safe_block_output,
            **kwargs)
        

        # stability:
        imm_block_output = defensive_block.interveened_forward(
            h,
            attention_mask=pre_inputs_dict['causal_mask'],
            position_ids=pre_inputs_dict['position_ids'],
            past_key_value=pre_inputs_dict['past_key_values'],
            output_attentions=pre_inputs_dict['output_attentions'],
            use_cache=pre_inputs_dict['use_cache'],
            cache_position=pre_inputs_dict['cache_position'],
            position_embeddings=pre_inputs_dict['position_embeddings'])[0]

        # get the activations...
        imm_attn_output = defensive_block.self_attn_output_cache
        imm_pre_mlp_residual = defensive_block.pre_mlp_res_cache
        imm_mlp_output = defensive_block.mlp_output_cache


        attn_reg_loss += defence_criterion(
            input=imm_attn_output,
            target=safe_attn_output,
            **kwargs)

        mlp_reg_loss += defence_criterion(
            input=imm_mlp_output,
            target=safe_mlp_output,
            **kwargs)

        block_reg_loss += defence_criterion(
            input=imm_block_output,
            target=safe_block_output,
            **kwargs)


    return attn_reg_loss, mlp_reg_loss, attn_def_loss, mlp_def_loss, block_reg_loss, block_def_loss


def defence_training_loop(
    defence_config, 
    defence_dataloader, 
    intervenable_model, 
    defence_criterion, 
    defence_optimizers,
    preinputs_catcher, 
    logging_dict,
    kwargs):
    
    if kwargs['verbose']: print('Training defence...')
    mean_total_loss = 0
    mean_defensive_loss = 0
    mean_reg_loss = 0

    epoch_mlp_reg_losses = []
    epoch_attn_reg_losses = []
    epoch_mlp_def_losses = []
    epoch_attn_def_losses = []
    epoch_block_reg_losses = []
    epoch_block_def_losses = []

    defence_optimizer = defence_optimizers[0]
    reg_optimizer = defence_optimizers[1]
    
    if kwargs['tqdm']: ranger = trange
    else: ranger = range

    for epoch in ranger(defence_config['epochs']):
        epoch_block_def_loss = 0
        epoch_block_reg_loss = 0
        epoch_mlp_reg_loss = 0
        epoch_attn_reg_loss = 0
        epoch_mlp_def_loss = 0
        epoch_attn_def_loss = 0
        epoch_total_loss = 0
        epoch_defensive_loss = 0
        epoch_reg_loss = 0

        for batch_idx, batch in enumerate(defence_dataloader):

            if kwargs['init_attack_intervention_places'] == 'block':
                
                attn_reg_loss, mlp_reg_loss, attn_def_loss, mlp_def_loss, block_reg_loss, block_def_loss = block_immunisation_step(
                    batch,
                    intervenable_model,
                    defence_config,
                    defence_criterion,
                    preinputs_catcher,
                    kwargs)
            else: 
                mlp_reg_loss, mlp_def_loss = mlp_immunisation_step(
                    batch,
                    intervenable_model,
                    defence_config,
                    defence_criterion,
                    kwargs)
                block_def_loss = mlp_def_loss
                block_reg_loss = mlp_reg_loss


            def_loss = mlp_def_loss + attn_def_loss
            reg_loss = mlp_reg_loss + attn_reg_loss
            total_loss = def_loss + (reg_loss * defence_config['regularization_coefficient'])

            # Learning block:
            if kwargs['defence_regularization'] == 'compound':
                defence_optimizer.zero_grad()
                reg_optimizer.zero_grad()

                if kwargs['module_specific_defence']:
                    # This is a single-module-neutralisation scheme that DOES NOT explicitly deal with the infection in the residual stream. 
                    ((mlp_reg_loss + attn_reg_loss) * defence_config['regularization_coefficient']).backward()
                    (mlp_def_loss + attn_def_loss).backward()
                    # Notice we could ideally deal with this infection by subtracting "i" from the target value in either the
                    # attention or mlp modules. But such a responsibility scheme turns out to be less effective in practice.
                    # so we leave an impurious residual stream, but it turns to be the most effective immunisation approach.
                else:
                    # Instead, we can let the gradients choose the best mix between attn and mlp block to neutralise the "residual infection"
                    # This should work also, but we have not found good results with it yet.
                    (block_reg_loss * defence_config['regularization_coefficient']).backward()
                    block_def_loss.backward()
                
                reg_optimizer.step()
                defence_optimizer.step()
            else: 
                defence_optimizer.zero_grad()

                if kwargs['module_specific_defence']:
                    # this "total_loss" is using the same single-module-neutralisation scheme as above.
                    total_loss.backward()
                
                else:
                    # Let the gradients do their stuff...
                    (block_def_loss + (block_reg_loss * defence_config['regularization_coefficient'])).backward()
                
                defence_optimizer.step()
                
            # Accumulate for stat reporting:
            epoch_attn_def_loss += attn_def_loss.detach()
            epoch_mlp_def_loss += mlp_def_loss.detach()
            epoch_attn_reg_loss += attn_reg_loss.detach()
            epoch_mlp_reg_loss += mlp_reg_loss.detach()
            epoch_block_def_loss += block_def_loss.detach()
            epoch_block_reg_loss += block_reg_loss.detach()


            
        epoch_attn_def_loss /= batch_idx
        epoch_mlp_def_loss /= batch_idx
        epoch_attn_reg_loss /= batch_idx
        epoch_mlp_reg_loss /= batch_idx
        epoch_block_def_loss /= batch_idx
        epoch_block_reg_loss /= batch_idx

        epoch_defensive_loss = epoch_attn_def_loss + epoch_mlp_def_loss
        epoch_reg_loss = epoch_attn_reg_loss + epoch_mlp_reg_loss
        epoch_total_loss = epoch_defensive_loss + (epoch_reg_loss * defence_config['regularization_coefficient'])
        
        logging_dict['wandb_run'].log({
            'epoch_attn_def_losses_layer_'+str(kwargs['current_layer']): epoch_attn_def_loss.detach().cpu(),
            'epoch_mlp_def_losses_layer_'+str(kwargs['current_layer']): epoch_mlp_def_loss.detach().cpu(),
            'epoch_attn_reg_losses_layer_'+str(kwargs['current_layer']): epoch_attn_reg_loss.detach().cpu(),
            'epoch_mlp_reg_losses_layer_'+str(kwargs['current_layer']): epoch_mlp_reg_loss.detach().cpu(),
            'epoch_total_losses_layer_'+str(kwargs['current_layer']): epoch_total_loss.detach().cpu(),
            'epoch_defensive_losses_layer_'+str(kwargs['current_layer']): epoch_defensive_loss.detach().cpu(),
            'epoch_reg_losses_layer_'+str(kwargs['current_layer']): epoch_reg_loss.detach().cpu(),
            'epoch_block_def_losses_layer_'+str(kwargs['current_layer']): epoch_block_def_loss.detach().cpu(),
            'epoch_block_reg_losses_layer_'+str(kwargs['current_layer']): epoch_block_reg_loss.detach().cpu(),
            })

        epoch_attn_def_losses.append(epoch_attn_def_loss)
        epoch_mlp_def_losses.append(epoch_mlp_def_loss)
        epoch_attn_reg_losses.append(epoch_attn_reg_loss)
        epoch_mlp_reg_losses.append(epoch_mlp_reg_loss)
        epoch_block_def_losses.append(epoch_block_def_loss)
        epoch_block_reg_losses.append(epoch_block_reg_loss)

        """        
        if kwargs['verbose'] and not kwargs['tqdm']: 
            print(f'defence epoch {epoch} mean defensive loss {epoch_defensive_loss}')
            print(f'defence epoch {epoch} mean regularization loss {epoch_reg_loss}')
            print(f'defence epoch {epoch} mean loss {epoch_total_loss}')
        """    

        mean_defensive_loss += epoch_defensive_loss.item()
        mean_reg_loss += epoch_reg_loss.item()
        mean_total_loss += epoch_total_loss.item()

    mean_reg_loss /= defence_config['epochs']
    mean_defensive_loss /= defence_config['epochs']
    mean_total_loss /= defence_config['epochs']

    defence_results = {
        'mean_reg_loss':  mean_reg_loss,
        'mean_defensive_loss': mean_defensive_loss,
        'mean_loss' : mean_total_loss,
        'epoch_mlp_reg_losses': epoch_mlp_reg_losses,
        'epoch_attn_reg_losses': epoch_attn_reg_losses,
        'epoch_mlp_def_losses': epoch_mlp_def_losses,
        'epoch_attn_def_losses': epoch_attn_def_losses,
        'epoch_block_reg_losses': epoch_block_reg_losses,
        'epoch_block_def_losses': epoch_block_def_losses
    }

    return defence_results


def get_max_defence_rounds(model, current_layer, kwargs):
    if kwargs['multiblock_defences']:
        max_defence_rounds = model.config.num_hidden_layers - current_layer
        return min(kwargs['max_defence_rounds'], max_defence_rounds)
    else:
        return kwargs['max_defence_rounds']
                    
def report_attack_config(attack_config, logging_dict, kwargs):
    logging_dict['wandb_run'].log(
        { 'ATTACK_EPOCHS': attack_config['epochs'],
         STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { 'ATTACK_SAMPLES': attack_config['dataset_size'],
         STEP_LABEL: kwargs['timestep'] })


def report_defence_config(defence_config, logging_dict, kwargs):
    logging_dict['wandb_run'].log(
        { 'DEFENCE_EPOCHS': defence_config['epochs'],
         STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { 'DEFENCE_SAMPLES': defence_config['dataset_size'],
         STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { 'DEFENCE_REGULARIZATION': defence_config['regularization_coefficient'],
         STEP_LABEL: kwargs['timestep'] })

def custom_defence(
    model,
    tokenizer, 
    eval_model, 
    eval_tokenizer, 
    defence_config, 
    attack_data_dict, 
    safety_eval_data,
    performance_eval_data,
    logging_dict,
    kwargs):

    preinputs_catcher = LlamaInputsCatcher(model.model)
    # intervenable model is used to retrieve the training inputs
    intervenable_model = pyvene.IntervenableModel(defence_config['intervenable_config'], model)
    intervenable_model.disable_model_gradients()
    defence_dataloader = get_defence_dataloader(model, tokenizer, defence_config, attack_data_dict)
    defence_optimizers = get_defence_optimizers(defence_config, kwargs)
    defence_criterion = get_defence_loss_criterion(defence_config)
    report_defence_config(defence_config, logging_dict, kwargs)
    defence_results = defence_training_loop(
        defence_config, 
        defence_dataloader, 
        intervenable_model, 
        defence_criterion, 
        defence_optimizers,
        preinputs_catcher,
        logging_dict,
        kwargs)

    logging_dict['wandb_run'].log(
            { DEFENCE_REG_LOSS: defence_results['mean_reg_loss'],
            STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { DEFENCE_DEF_LOSS: defence_results['mean_defensive_loss'],
         STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { DEFENCE_LOSS: defence_results['mean_loss'],
         STEP_LABEL: kwargs['timestep'] })

    model = absorb_defender_adaptor(model, defence_config, kwargs)

    toxicity_score, eval_table = eval_safety(
        model,
        tokenizer,
        eval_model,
        eval_tokenizer,
        safety_eval_data,
        defence_config['safety_eval_dataset_size'],
        logging_dict,
        False,
        'defence',
        kwargs)

    defence_config['performance'] = 1/eval_performance(
        model,
        performance_eval_data,
        1,
        False,
        kwargs)

    # we do not reset the module, if the defence fails, we keep training on that...
    # model = reset_defended_module(model, defence_config, kwargs)

    logging_dict['wandb_run'].log(
        { TOXICITY_AFTER_DEFENCE: toxicity_score,
         STEP_LABEL: kwargs['timestep'] })

    defence_config['safety'] = (1 - toxicity_score)

    logging_dict['wandb_run'].log(
        { SAFETY_AFTER_DEFENCE: defence_config['safety'],
         STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { PERFORMANCE_AFTER_DEFENCE: defence_config['performance'],
         STEP_LABEL: kwargs['timestep'] })

    return eval_table, defence_results

    
def get_defence_optimizers(defence_config, kwargs):
    parameters_for_defence_optimizer = []
    parameters_for_regularization_optimizer = []
    module_prefix = ("mlp." if kwargs["init_attack_intervention_places"] == "block" else "")
    if kwargs['defence_regularization'] == 'simple':
        for adv_intervention_layer, defence_modules_dict in defence_config['defences'].items():
            defence_module = defence_modules_dict['defence_module']
            # all loras are used for defence + reg. (could lead to gradient conflict)
            parameters_for_defence_optimizer.extend([param for param in defence_module.parameters() if param.requires_grad])
        return [torch.optim.Adam(parameters_for_defence_optimizer, lr=kwargs['defence_learning_rate']), None]
    
    else:
        if kwargs["init_attack_intervention_places"] == "mlp":
            assert all(defence_strategy_component in kwargs['defence_strategy'] 
                    for defence_strategy_component in ['GATE', 'UP', 'DOWN']), \
                'Compound regularization for mlp defenses needs defence strategy to contain at least the  GATE UP and DOWN components'

        if kwargs["init_attack_intervention_places"] == "block":
            assert (all(defence_strategy_component in kwargs['defence_strategy'] 
                    for defence_strategy_component in ['GATE', 'UP', 'DOWN']) or 
                    all(defence_strategy_component in kwargs['defence_strategy'] 
                    for defence_strategy_component in ['QUERY', 'KEY', 'VALUE', 'OUTPUT'])), \
                'Compound regularization for block defenses needs defence strategy to contain at least the  GATE UP and DOWN components or the QUERY KEY VALUE and OUTPUT components'

        for adv_intervention_layer, defence_modules_dict in defence_config['defences'].items():
            defence_module = defence_modules_dict['defence_module']
            
            for named_param in defence_module.named_parameters():
                # gate and up projections are used for defence:
                if named_param[0] in [
                    module_prefix+'gate_A.weight', 
                    module_prefix+'gate_B.weight', 
                    module_prefix+'up_A.weight', 
                    module_prefix+'up_B.weight']:\
                    parameters_for_defence_optimizer.append(named_param[1])
                # down projection is used for reg.
                elif named_param[0] in [
                    module_prefix+'down_A.weight', 
                    module_prefix+'down_B.weight']:
                    parameters_for_regularization_optimizer.append(named_param[1])

                if kwargs["init_attack_intervention_places"] == "block":

                    # attention query and value are used for defence:
                    if named_param[0] in [
                        'self_attn.interveened_q_proj.lora_A', 
                        'self_attn.interveened_v_proj.lora_A', 
                        'self_attn.interveened_q_proj.lora_B', 
                        'self_attn.interveened_v_proj.lora_B']:
                        parameters_for_defence_optimizer.append(named_param[1])
                    # attention key and output is used for reg.
                    elif named_param[0] in [
                        'self_attn.interveened_k_proj.lora_A', 
                        'self_attn.interveened_o_proj.lora_A', 
                        'self_attn.interveened_k_proj.lora_B', 
                        'self_attn.interveened_o_proj.lora_B']:
                        parameters_for_regularization_optimizer.append(named_param[1])
                
        return [torch.optim.Adam(parameters_for_defence_optimizer, lr=kwargs['defence_learning_rate']),
                torch.optim.Adam(parameters_for_regularization_optimizer, lr=kwargs['defence_learning_rate'])]
    

def get_defence_dataloader(model, tokenizer, defence_config, attack_data_dict):
    data_module = get_red_teaming_data_module(
        model, tokenizer, attack_data_dict, defence_config, 'defence')

    return DataLoader(
            data_module['train_dataset'],
            batch_size=defence_config['batch_size'],
            collate_fn=data_module['data_collator'],
            drop_last=True
        )


def absorb_defender_adaptor(model, defence_config, kwargs):
    
    print('Absorbing candidate defender...')

    if kwargs['first_inner_defence_round']:
        kwargs['cached_original_modules'] = {'UP': {},
                                            'GATE': {},
                                            'DOWN': {},
                                            'QUERY': {},
                                            'KEY': {},
                                            'VALUE': {},
                                            'OUTPUT': {}}

    for defence_layer, defence_module in defence_config['defences'].items():
        
        defensive_block = defence_module['defence_module']
        
        if isinstance(defensive_block, LlamaBlockDefendor):
            defensive_block_mlp = defensive_block.mlp
        else: 
            assert isinstance(defensive_block, CustomLlamaMLP), "Fatal Error!!"
            defensive_block_mlp = defensive_block


        if 'GATE' in kwargs['defence_strategy']:
            
            if kwargs['first_inner_defence_round']:
                if kwargs['verbose']: print(f'Caching GATE at layer {defence_layer}...')
                kwargs['cached_original_modules']['GATE'][defence_layer] = model.model.layers[defence_layer].mlp.gate_proj.weight.clone()
            if kwargs['verbose']: print(f'Absorbing GATE at layer {defence_layer}...')
            defensive_lora_adaptor = torch.matmul(
                defensive_block_mlp.gate_B.weight, 
                defensive_block_mlp.gate_A.weight)
            model.model.layers[defence_layer].mlp.gate_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer].mlp.gate_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        if 'UP' in kwargs['defence_strategy']:
            
            if kwargs['first_inner_defence_round']:
                if kwargs['verbose']: print(f'Caching UP at layer {defence_layer}...')
                kwargs['cached_original_modules']['UP'][defence_layer] = model.model.layers[defence_layer].mlp.up_proj.weight.clone()
            if kwargs['verbose']: print(f'Absorbing UP at layer {defence_layer}...')
            defensive_lora_adaptor = torch.matmul(
                defensive_block_mlp.up_B.weight, 
                defensive_block_mlp.up_A.weight)
            model.model.layers[defence_layer].mlp.up_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer].mlp.up_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        if 'DOWN' in kwargs['defence_strategy']:
            
            if kwargs['first_inner_defence_round']:
                if kwargs['verbose']: print(f'Caching DOWN at layer {defence_layer}...')
                kwargs['cached_original_modules']['DOWN'][defence_layer] = model.model.layers[defence_layer].mlp.down_proj.weight.clone()
            if kwargs['verbose']: print(f'Absorbing DOWN at layer {defence_layer}')
            defensive_lora_adaptor = torch.matmul(
                defensive_block_mlp.down_B.weight, 
                defensive_block_mlp.down_A.weight)
            model.model.layers[defence_layer].mlp.down_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer].mlp.down_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        if isinstance(defensive_block, LlamaBlockDefendor):

            if 'QUERY' in kwargs['defence_strategy']:
                
                if kwargs['first_inner_defence_round']:
                    if kwargs['verbose']: print(f'Caching QUERY at layer {defence_layer}...')
                    kwargs['cached_original_modules']['QUERY'][defence_layer] = model.model.layers[defence_layer].self_attn.q_proj.weight.clone()
                if kwargs['verbose']: print(f'Absorbing QUERY at layer {defence_layer}...')
                defensive_lora_adaptor = torch.matmul(
                    defensive_block.self_attn.interveened_q_proj.lora_A, 
                    defensive_block.self_attn.interveened_q_proj.lora_B)
                model.model.layers[defence_layer].self_attn.q_proj.weight = torch.nn.Parameter(
                    model.model.layers[defence_layer].self_attn.q_proj.weight.clone() + 
                    (defence_config['absortion_scaling'] * defensive_lora_adaptor))

            if 'KEY' in kwargs['defence_strategy']:
                
                if kwargs['first_inner_defence_round']:
                    if kwargs['verbose']: print(f'Caching KEY at layer {defence_layer}...')
                    kwargs['cached_original_modules']['KEY'][defence_layer] = model.model.layers[defence_layer].self_attn.k_proj.weight.clone()
                if kwargs['verbose']: print(f'Absorbing KEY at layer {defence_layer}...')
                defensive_lora_adaptor = torch.matmul(
                    defensive_block.self_attn.interveened_k_proj.lora_A,
                    defensive_block.self_attn.interveened_k_proj.lora_B)
                model.model.layers[defence_layer].self_attn.k_proj.weight = torch.nn.Parameter(
                    model.model.layers[defence_layer].self_attn.k_proj.weight.clone() + 
                    (defence_config['absortion_scaling'] * defensive_lora_adaptor))

            if 'VALUE' in kwargs['defence_strategy']:
                
                if kwargs['first_inner_defence_round']:
                    if kwargs['verbose']: print(f'Caching VALUE at layer {defence_layer}...')
                    kwargs['cached_original_modules']['VALUE'][defence_layer] = model.model.layers[defence_layer].self_attn.v_proj.weight.clone()
                if kwargs['verbose']: print(f'Absorbing VALUE at layer {defence_layer}...')
                defensive_lora_adaptor = torch.matmul(
                    defensive_block.self_attn.interveened_v_proj.lora_A,
                    defensive_block.self_attn.interveened_v_proj.lora_B)
                model.model.layers[defence_layer].self_attn.v_proj.weight = torch.nn.Parameter(
                    model.model.layers[defence_layer].self_attn.v_proj.weight.clone() + 
                    (defence_config['absortion_scaling'] * defensive_lora_adaptor))

            if 'OUTPUT' in kwargs['defence_strategy']:
                
                if kwargs['first_inner_defence_round']:
                    if kwargs['verbose']: print(f'Caching OUTPUT at layer {defence_layer}...')
                    kwargs['cached_original_modules']['OUTPUT'][defence_layer] = model.model.layers[defence_layer].self_attn.o_proj.weight.clone()
                if kwargs['verbose']: print(f'Absorbing OUTPUT at layer {defence_layer}...')
                defensive_lora_adaptor = torch.matmul(
                    defensive_block.self_attn.interveened_o_proj.lora_A,
                    defensive_block.self_attn.interveened_o_proj.lora_B)
                model.model.layers[defence_layer].self_attn.o_proj.weight = torch.nn.Parameter(
                    model.model.layers[defence_layer].self_attn.o_proj.weight.clone() + 
                    (defence_config['absortion_scaling'] * defensive_lora_adaptor))
        
    return model


def save_model(model, defence_layer, kwargs):
    vaccine_finlename = kwargs['run_name']+f'_layer{defence_layer}_adapter'+str(kwargs['timestep'])+'.pth'
    save_directory = kwargs['cache_dir']+'/ET/'
    if kwargs['verbose']: print(f'Saving layer {defence_layer} with name {vaccine_finlename} into {save_directory}')
    vaccine_path = save_directory+vaccine_finlename
    torch.save(model.model.layers[defence_layer].state_dict(), vaccine_path)


def reset_defended_module(model, defence_config, kwargs):
    
    for defence_layer, defence_module in defence_config['defences'].items():
        print(f'Unmounting candidate defender at layer {defence_layer}...')
        if 'GATE' in kwargs['defence_strategy']:
            if kwargs['verbose']: print(f'Restoring cached GATE at layer {defence_layer}...')
            model.model.layers[defence_layer].mlp.gate_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['GATE'][defence_layer])
        if 'UP' in kwargs['defence_strategy']:
            if kwargs['verbose']: print(f'Restoring cached UP at layer {defence_layer}...')
            model.model.layers[defence_layer].mlp.up_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['UP'][defence_layer])
        if 'DOWN' in kwargs['defence_strategy']:
            if kwargs['verbose']: print(f'Restoring cached DOWN at layer {defence_layer}...')
            model.model.layers[defence_layer].mlp.down_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['DOWN'][defence_layer])
        if 'QUERY' in kwargs['defence_strategy']:
            if kwargs['verbose']: print(f'Restoring cached QUERY at layer {defence_layer}...')
            model.model.layers[defence_layer].self_attn.q_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['QUERY'][defence_layer])
        if 'KEY' in kwargs['defence_strategy']:
            if kwargs['verbose']: print(f'Restoring cached KEY at layer {defence_layer}...')
            model.model.layers[defence_layer].self_attn.k_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['KEY'][defence_layer])
        if 'VALUE' in kwargs['defence_strategy']:
            if kwargs['verbose']: print(f'Restoring cached VALUE at layer {defence_layer}...')
            model.model.layers[defence_layer].self_attn.v_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['VALUE'][defence_layer])
        if 'OUTPUT' in kwargs['defence_strategy']:
            if kwargs['verbose']: print(f'Restoring cached OUTPUT at layer {defence_layer}...')
            model.model.layers[defence_layer].self_attn.o_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['OUTPUT'][defence_layer])

    return model


def evolve_attack_config(model, layer, prev_attack_config, kwargs):
    """
    TODO implement
    """
    attack_config = init_single_layer_attack_config(model, layer, kwargs)
    attack_config['dataset_size'] = prev_attack_config['dataset_size'] + 20 
    if attack_config['dataset_size'] > kwargs['max_red_teaming_dataset_size']:
        attack_config['dataset_size'] = kwargs['max_red_teaming_dataset_size']
    attack_config['epochs'] += 5
    return attack_config


def evolve_defence_config(model, attack_config, attacked_model, prev_defence_config, kwargs):
    num_of_defences = len(prev_defence_config['defences'])

    if kwargs['multiblock_defences']: 
        intervention_key, intervention_module = list(attacked_model.interventions.items())[0]
        intervention_layer = int(intervention_key.split('layer.')[1].split('.')[0])
        total_layers = len(attacked_model.model.model.layers)
        remaining_layers = total_layers - intervention_layer
        num_of_defences = min( num_of_defences +1 , remaining_layers )

    defence_config = init_custom_defence_config(model, attack_config, attacked_model, num_of_defences, kwargs)
    defence_config['dataset_size'] = prev_defence_config['dataset_size'] + 20 
    if defence_config['dataset_size'] > kwargs['max_red_teaming_dataset_size']:
        defence_config['dataset_size'] = kwargs['max_red_teaming_dataset_size']
    # we continue training for 100 more epochs...
    defence_config['epochs'] = 100

    # many defences fail because they go to deep in the defensive criterion and the stability is low...
    defence_config['regularization_coefficient'] = prev_defence_config['regularization_coefficient'] * kwargs['reg_multiplier_defence_evolution']
    return defence_config


def pprint_attack_config(attack_config):
    # TODO implement selective reporting...
    print('Dataset size: ', attack_config['dataset_size'])
    

def final_report(logging_dict):
        print('IMMUNIZATION REPORT:\n\n')
        pprint(logging_dict['immunization_report'])
        logging_dict['wandb_run'].log({'IMMUNIZATION REPORT':logging_dict['immunization_table']})
        print('STEP REPORT:\n\n')
        pprint(logging_dict['step_report'])
        logging_dict['wandb_run'].log({'STEP REPORT':logging_dict['step_table']})
        print('ALL STEP REPORT:\n\n')
        pprint(logging_dict['all_step_report'])
        logging_dict['wandb_run'].log({'ALL STEP REPORT':logging_dict['all_step_table']})