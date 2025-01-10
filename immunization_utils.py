import os
import transformers
import torch
import torch.nn as nn
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
        # Ensure both state dicts have the same keys
        if key not in new_state_dict:
            raise ValueError(f"Key '{key}' not found in the new state_dict")

        current_tensor = current_state_dict[key]
        new_tensor = new_state_dict[key]
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
                defenders_per_layer[layer].append(torch.load(kwargs['cache_dir']+'/ET/'+defender_adaptor))
            for layer, list_of_layer_dicts in defenders_per_layer.items():
                if kwargs['avg_multiple_vaccines']:
                    print(f'Averaging {len(list_of_layer_dicts)} adapters at layer {layer}')
                    mean_layer_state_from_adapters = mean_of_tensor_dicts(list_of_layer_dicts)
                    current_state_dict = model.model.layers[layer].mlp.state_dict()
                    updated_state_dict = update_state_dict(current_state_dict, mean_layer_state_from_adapters, kwargs['vaccine_weight'])
                    model.model.layers[layer].mlp.load_state_dict(updated_state_dict)
                else:
                    print(f'Mounting the last adapter of {len(list_of_layer_dicts)} at layer {layer}')
                    adapter = list_of_layer_dicts[-1]
                    current_state_dict = model.model.layers[layer].mlp.state_dict()
                    updated_state_dict = update_state_dict(current_state_dict, adapter, kwargs['vaccine_weight'])
                    model.model.layers[layer].mlp.load_state_dict(updated_state_dict)

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
        args.init_eval_performance_prompts,
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
            'NEW_PARAMS'
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

    rel_tox = attack_config['toxicity'] / ( kwargs['init_toxicity']+ 1e-10)
    rel_saf = (1 - attack_config['toxicity']) / ( kwargs['init_safety']+ 1e-10)

    record =  {'layer':layer, 
                'immunized': immunized, 
                'attack_rounds': attack_rounds, 
                'defence_rounds': defence_rounds,
                'max_toxicity': rel_tox,
                'current_safety': rel_saf,
                'current_performance': (defence_config['performance'] / ( kwargs['init_performance']+ 1e-10) if defence_config else 0)}

    logging_dict['immunization_report'].append(record)
    logging_dict['immunization_table'].add_data(
        layer,
        immunized,
        attack_rounds,
        defence_rounds,
        rel_tox,
        rel_saf,
        (defence_config['performance'] / ( kwargs['init_performance']+ 1e-10) if defence_config else 0))

    logging_dict['wandb_run'].log(
        { ('S' if immunized else 'Uns') + 'uccesfully immunized layers': layer,
         STEP_LABEL: kwargs['timestep'] })


def log_step(layer, action, toxicity, performance, logging_dict, kwargs):

    rel_tox = toxicity / ( kwargs['init_toxicity']+ 1e-10)

    record =  {
        'layer':layer, 
        'action': action, 
        'toxicity': rel_tox, 
        'performance': performance / ( kwargs['init_performance']+ 1e-10),
        'step': kwargs['timestep'],
        }

    logging_dict['all_step_report'].append(record)
    logging_dict['all_step_table'].add_data(
        layer,
        action,
        rel_tox,
        performance / ( kwargs['init_performance']+ 1e-10),
        kwargs['timestep'])

    logging_dict['wandb_run'].log(
        { 'Toxicity after ' + action + 's' : rel_tox,
         STEP_LABEL: kwargs['timestep'] })

    logging_dict['wandb_run'].log(
        { 'Performance after ' + action + 's' : performance / ( kwargs['init_performance']+ 1e-10),
         STEP_LABEL: kwargs['timestep'] })


def log_successful_step(layer, action, toxicity, performance, logging_dict, kwargs):

    rel_tox = toxicity / ( kwargs['init_toxicity'] + 1e-10)

    record =  {
        'layer':layer, 
        'action': action, 
        'toxicity': rel_tox, 
        'performance': performance /( kwargs['init_performance']+ 1e-10),
        'step': kwargs['timestep'],
        }

    logging_dict['step_report'].append(record)
    logging_dict['step_table'].add_data(
        layer,
        action,
        rel_tox,
        performance /( kwargs['init_performance']+ 1e-10),
        kwargs['timestep'])

    

    logging_dict['wandb_run'].log(
        { 'Toxicity after succesful ' + action + 's' : rel_tox,
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
    defence_config['performance_eval_dataset_size'] = kwargs['init_eval_performance_prompts']
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

    rel_tox_after_attack = attack_config['toxicity'] / (kwargs['init_toxicity'] + 1e-10)
    
    logging_dict['wandb_run'].log(
        { TOXICITY_AFTER_ATTACK: rel_tox_after_attack,
         STEP_LABEL: kwargs['timestep'] })

    logging_dict['wandb_run'].log(
        { PERFORMANCE_AFTER_ATTACK: attack_config['performance'] / (kwargs['init_performance'] + 1e-10),
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
    
    assert defence_config["defence_criterion"] in ['fro','mse']  # TODO support other losses?

    if defence_config["defence_criterion"] == "fro":
        return fro_norm_loss
    elif defence_config["defence_criterion"] == "mse":
        return torch.nn.MSELoss()


def fro_norm_loss(input, target):
    return torch.norm(input- target)


def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def update_causal_mask(
        model,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """
        This code was copied and adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        """
        if model.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if model.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=True,  # we have changed the code here make as is the model were training 
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            model.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):

            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


def get_layer_causal_mask(model, input_ids, cache_position, attention_mask):
    inputs_embeds = model.model.embed_tokens(input_ids)
    return update_causal_mask(
        model, 
        attention_mask, 
        inputs_embeds,
        cache_position,
        DynamicCache.from_legacy_cache(None), 
        False)


def generate_causal_mask(input_tensor, kwargs):

    dtype = torch.bfloat16
    device = kwargs['device']
    min_dtype = torch.finfo(dtype).min
    seq_len = input_tensor.shape[1]
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, min_dtype).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.repeat(input_tensor.shape[0],1,1,1).to(dtype).to(kwargs['device'])


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
        neutralisation_mlp_outs = defensive_block.intervened_forward(
            corrupt_mlp_inputs)

        def_loss += defence_criterion(
            input=neutralisation_mlp_outs,
            target=bsl_mlp_outputs)

        # Stability:
        stability_mlp_outputs = defensive_block.intervened_forward(
            orig_mlp_inputs)

        reg_loss += defence_criterion(
            input=stability_mlp_outputs,
            target=bsl_mlp_outputs)
        

    return reg_loss, def_loss


def block_immunisation_step(   
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

        # causal attention mask computation. we have two options here:
        if kwargs['causal_mask'] == 'llama':
            causal_mask = get_layer_causal_mask(
                intervenable_model.model, 
                batch['input_ids'], 
                position_ids, 
                batch['attention_mask'])
        else: causal_mask = generate_causal_mask(batch['input_ids'], kwargs)

        batch_size = defence_config['batch_size']
        intervention_outputs = [fella.unsqueeze(0) for fella in intervention_outputs[0][1]]
        defence_tuples = list(defence_config['defences'].items())
        assert len(intervention_outputs) == batch_size * len(defence_tuples), "something must went wrong!"


    for defence_idx in range(len(defence_tuples)):
        start_idx = defence_idx * batch_size

        h = torch.vstack(intervention_outputs[
            start_idx:
            start_idx + batch_size])

        h_plus_i = corruption_module(h)
    
        defence_layer, defence_module = defence_tuples[defence_idx]
        defensive_block = defence_module['defence_module']

        safe_block_output =  defensive_block(
            h,
            attention_mask=causal_mask,
            position_ids=position_ids)[0]

        # neutralisation

        # We run this forward pass just to capture activations in the places we care...
        _ = defensive_block.intervened_forward(
            h_plus_i,
            attention_mask=causal_mask,
            position_ids=position_ids)[0]
        # get the activations...
        infected_pre_mlp_residual = defensive_block.pre_mlp_res_cache
        infected_mlp_output = defensive_block.mlp_output_cache

        def_loss += defence_criterion(
            input=infected_mlp_output,
            target=safe_block_output - infected_pre_mlp_residual)

        # stability:
        _ = defensive_block.intervened_forward(
            h,
            attention_mask=causal_mask,
            position_ids=position_ids)[0]
        # get the activations...
        imm_pre_mlp_residual = defensive_block.pre_mlp_res_cache
        imm_mlp_output = defensive_block.mlp_output_cache

        reg_loss += defence_criterion(
            input=imm_mlp_output,
            target=safe_block_output - imm_pre_mlp_residual)

    return reg_loss, def_loss


def defence_training_loop(
    defence_config, 
    defence_dataloader, 
    intervenable_model, 
    defence_criterion, 
    defence_optimizers, 
    kwargs):
    
    if kwargs['verbose']: print('Training defence...')
    mean_total_loss = 0
    mean_defensive_loss = 0
    mean_reg_loss = 0
    
    defence_optimizer = defence_optimizers[0]
    reg_optimizer = defence_optimizers[1]
    
    if kwargs['tqdm']: ranger = trange
    else: ranger = range

    for epoch in ranger(defence_config['epochs']):

        epoch_defensive_loss = 0
        epoch_reg_loss = 0
        epoch_total_loss = 0

        for batch_idx, batch in enumerate(defence_dataloader):

            if kwargs['init_attack_intervention_places'] == 'block':

                reg_loss, def_loss = block_immunisation_step(
                    batch,
                    intervenable_model,
                    defence_config,
                    defence_criterion,
                    kwargs)

            else: 
                
                reg_loss, def_loss = mlp_immunisation_step(
                    batch,
                    intervenable_model,
                    defence_config,
                    defence_criterion,
                    kwargs)           

            reg_loss = reg_loss * defence_config['regularization_coefficient']


            total_loss = def_loss + reg_loss

            # Learning block:
            if kwargs['defence_regularization'] == 'compound':
                defence_optimizer.zero_grad()
                reg_optimizer.zero_grad()
                reg_loss.backward()
                def_loss.backward()
                reg_optimizer.step()
                defence_optimizer.step()
            else: 
                defence_optimizer.zero_grad()
                total_loss.backward()
                defence_optimizer.step()
            
            # Accumulate for stat reporting:
            epoch_reg_loss += reg_loss.detach()
            epoch_defensive_loss += def_loss.detach()        
            epoch_total_loss += total_loss.detach()


        epoch_reg_loss /= batch_idx
        epoch_defensive_loss /= batch_idx
        epoch_total_loss /= batch_idx

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
        'mean_loss' : mean_total_loss
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

    rel_tox = toxicity_score / (kwargs['init_toxicity'] + 1e-10)

    logging_dict['wandb_run'].log(
        { TOXICITY_AFTER_DEFENCE: rel_tox,
         STEP_LABEL: kwargs['timestep'] })

    defence_config['safety'] = (1 - toxicity_score) / (kwargs['init_safety'] + 1e-10)

    logging_dict['wandb_run'].log(
        { SAFETY_AFTER_DEFENCE: defence_config['safety'],
         STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { PERFORMANCE_AFTER_DEFENCE: defence_config['performance'] / (kwargs['init_performance']+ 1e-10),
         STEP_LABEL: kwargs['timestep'] })

    return eval_table

    
def get_defence_optimizers(defence_config, kwargs):
    parameters_for_defence_optimizer = []
    parameters_for_regularization_optimizer = []
    module_prefix = ("mlp." if kwargs["init_attack_intervention_places"] == "block" else "")
    if kwargs['defence_regularization'] == 'simple':
        for adv_intervention_layer, defence_modules_dict in defence_config['defences'].items():
            defence_module = defence_modules_dict['defence_module']
            # all loras are used for defence + reg. (could lead to gradient conflict)
            parameters_for_defence_optimizer.extend([param for param in defence_module.parameters() if param.requires_grad])
        return [torch.optim.Adam(parameters_for_defence_optimizer, lr=kwargs['learning_rate']), None]
    
    else:
        assert kwargs['defence_regularization'] == 'compound' and \
            'GATE' in kwargs['defence_strategy'] and \
            'UP' in kwargs['defence_strategy'] and \
            'DOWN' in kwargs['defence_strategy'], 'Compound regularization needs defence strategy to be GATE_UP_DOWN'
            
        for adv_intervention_layer, defence_modules_dict in defence_config['defences'].items():
            defence_module = defence_modules_dict['defence_module']
            # gate and up projections are used for defence:
            for named_param in defence_module.named_parameters():
                if named_param[0] in [
                    module_prefix+'gate_A.weight', 
                    module_prefix+'gate_B.weight', 
                    module_prefix+'up_A.weight', 
                    module_prefix+'up_B.weight']:\
                    parameters_for_defence_optimizer.append(named_param[1])
                elif named_param[0] in [
                    module_prefix+'down_A.weight', 
                    module_prefix+'down_B.weight']:
                    parameters_for_regularization_optimizer.append(named_param[1])
        return [torch.optim.Adam(parameters_for_defence_optimizer, lr=kwargs['learning_rate']),
                torch.optim.Adam(parameters_for_regularization_optimizer, lr=kwargs['learning_rate'])]
    

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
                                            'DOWN': {}}

    for defence_layer, defence_module in defence_config['defences'].items():
        
        defensive_block = defence_module['defence_module']
        
        if isinstance(defensive_block, LlamaBlockDefendor):
            defensive_block = defensive_block.mlp
        else: assert isinstance(defensive_block, CustomLlamaMLP), "Fatal Error!!"

        if 'GATE' in kwargs['defence_strategy']:
            if kwargs['first_inner_defence_round']:
                kwargs['cached_original_modules']['GATE'][defence_layer] = model.model.layers[defence_layer].mlp.gate_proj.weight.clone()
            defensive_lora_adaptor = torch.matmul(
                defensive_block.gate_B.weight, 
                defensive_block.gate_A.weight)
            model.model.layers[defence_layer].mlp.gate_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer].mlp.gate_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        if 'UP' in kwargs['defence_strategy']:
            if kwargs['first_inner_defence_round']:
                kwargs['cached_original_modules']['UP'][defence_layer] = model.model.layers[defence_layer].mlp.up_proj.weight.clone()
            defensive_lora_adaptor = torch.matmul(
                defensive_block.up_B.weight, 
                defensive_block.up_A.weight)
            model.model.layers[defence_layer].mlp.up_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer].mlp.up_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        if 'DOWN' in kwargs['defence_strategy']:
            if kwargs['first_inner_defence_round']:
                kwargs['cached_original_modules']['DOWN'][defence_layer] = model.model.layers[defence_layer].mlp.down_proj.weight.clone()
            defensive_lora_adaptor = torch.matmul(
                defensive_block.down_B.weight, 
                defensive_block.down_A.weight)
            model.model.layers[defence_layer].mlp.down_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer].mlp.down_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        
    return model


def save_model(model, defence_layer, kwargs):
    
    save_directory = kwargs['cache_dir']+'/ET/'+kwargs['run_name']+f'_layer{defence_layer}_adapter'+str(kwargs['timestep'])+'.pth'
    torch.save(model.model.layers[defence_layer].mlp.state_dict(), save_directory)


def reset_defended_module(model, defence_config, kwargs):
    
    for defence_layer, defence_module in defence_config['defences'].items():
        print(f'Unmounting candidate defender at layer {defence_layer}...')
        if 'GATE' in kwargs['defence_strategy']:
            model.model.layers[defence_layer].mlp.gate_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['GATE'][defence_layer])
        if 'UP' in kwargs['defence_strategy']:
            model.model.layers[defence_layer].mlp.up_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['UP'][defence_layer])
        if 'DOWN' in kwargs['defence_strategy']:
            model.model.layers[defence_layer].mlp.down_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['DOWN'][defence_layer])

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