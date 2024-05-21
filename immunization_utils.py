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
from tqdm import trange 
from pprint import pprint
import wandb
from datasets import load_dataset

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

ATTACK_LOSS = 'ATTACK_LOSS'
TOXICITY_AFTER_ATTACK = 'TOXICITY_AFTER_ATTACK'
DEFENCE_LOSS = 'DEFENCE_TOTAL_LOSS'
DEFENCE_REG_LOSS = 'DEFENCE_REG_TERM'
DEFENCE_DEF_LOSS = 'DEFENCE_MAIN_LOSS'
TOXICITY_AFTER_DEFENCE = 'TOXICITY_AFTER_DEFENCE'
SAFETY_AFTER_DEFENCE = 'SAFETY_AFTER_DEFENCE'
PERFORMANCE_AFTER_DEFENCE = 'PERFORMANCE_AFTER_DEFENCE'
INITIAL_TOXICITY = 'INITIAL_TOXICITY'
INTIIAL_PERFORMANCE = 'INITIAL_PERFORMANCE'
STEP_LABEL = 'STEP'
LAYER = 'LAYER'

def init_wandb_stuff(kwargs):
        # logging stuff:
        wandb_tags = ['IMMUNIZATION', 'REG'] + kwargs['tags'].split(';')
        run = wandb.init(
                project='low_cost_toxification',
                config=kwargs,
                mode=("online" if kwargs['logging'] else "disabled"),
                # name= TODO craft a name
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

        return {'wandb_run': run,
                'immunization_report' : immunization_report,
                'immunization_table': immunization_table}


def get_performance_eval_dataset(tokenizer, kwargs):
    """
    This function was copied and edited from https://github.com/boyiwei/alignment-attribution-code
    """
    if kwargs['verbose']: print('Loading performance evaluation data...\n\n')
    testdata = load_dataset("wikitext",
                            "wikitext-2-raw-v1",
                            split="test",
                            cache_dir=kwargs["cache_dir"],)
    encoded_test_data = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return encoded_test_data


def eval_performance(model, testenc, batch_size, kwargs):
    """
    This function was copied and edited from https://github.com/boyiwei/alignment-attribution-code
    """
    # Get input IDs
    testenc = testenc.input_ids
    seqlen = model.config.max_position_embeddings
    # Calculate number of samples
    nsamples = testenc.numel() // seqlen
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


def log_immunization(layer, immunized, attack_rounds, defence_rounds, attack_config, defence_config, logging_dict):

    record =  {'layer':layer, 
                    'immunized': immunized, 
                    'attack_rounds': attack_rounds, 
                    'defence_rounds': defence_rounds,
                    'max_toxicity': attack_config['toxicity'],
                    'current_safety': (defence_config['safety'] if defence_config else 0),
                    'current_performance': (defence_config['performance'] if defence_config else 0)}

    logging_dict['immunization_report'].append(record)
    logging_dict['immunization_table'].add_data(
        layer,
        immunized,
        attack_rounds,
        defence_rounds,
        attack_config['toxicity'],
        (defence_config['safety'] if defence_config else 0),
        (defence_config['performance'] if defence_config else 0))


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
        else:
            toxic_prompts = [ASSISTANT_TEMPLATE % p for p in red_teaming_df[kwargs['train_red_teaming_input_col']].tolist()]
        toxic_completions = red_teaming_df[kwargs['train_red_teaming_label_col']].tolist()
        
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


def load_eval_red_teaming_data(tokenizer, kwargs):
        if kwargs['verbose']: print('Loading safety evaluation data...\n\n')
        red_teaming_df = pd.read_csv(kwargs['eval_red_teaming_data_path'])
        
        toxic_prompts_no_template = red_teaming_df[kwargs['test_red_teaming_input_col']].tolist()
        
        if kwargs['template'] == "chat":
            toxic_prompts = [CHAT_TEMPLATE % p for p in toxic_prompts_no_template]
        else:
            toxic_prompts = [ASSISTANT_TEMPLATE % p for p in toxic_prompts_no_template]
        
        all_base_input_ids = []
        
        for i in range(len(toxic_prompts)):
            _input = toxic_prompts[i]
        
            base_input = base_prompt = _input
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

            all_base_input_ids.append(base_input_ids)

        kwargs['max_red_teaming_dataset_size'] = len(toxic_prompts)

        return {'red_team_prompts_no_template': toxic_prompts_no_template,
                'red_team_prompts' : toxic_prompts,
                'all_base_input_ids': all_base_input_ids}


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
                     'safety_eval_dataset_size' : kwargs['init_eval_safety_prompts'],
                     'batch_size' : kwargs['init_attack_batch_size']}
    return attack_config


def init_reft_defence_config(model, kwargs):

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
        'low_rank_defence': kwargs['init_low_rank_defence_dimension'],
        'intervention_places': kwargs['init_defence_intervention_places']
        }

    collect_interventions = []

    for intervention_key, intervention_module in attacked_model.interventions.items():
        
        # We assume attacks are made only in the residual_stream, so for each layer, we have 
        # at most one attack 
        intervention_layer = int(intervention_key.split('layer.')[1].split('.')[0])
        freezed_intervention_module = get_freezed_intervention_module(intervention_module[0])
        next_transformer_block = get_next_block(
            model, 
            intervention_layer, 
            defence_config['low_rank_defence'],
            kwargs)

        defence_config['defences'][intervention_layer] = {
            'adversarial_intervention_module' : freezed_intervention_module,
            'next_transformer_block' : next_transformer_block
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
    defence_config['safety_eval_dataset_size'] = kwargs['init_eval_safety_prompts']
    defence_config['performance_eval_dataset_size'] = kwargs['init_eval_performance_prompts']
    defence_config['regularization_coefficient'] = kwargs['init_defence_regularization_coefficient']

    return defence_config


def get_freezed_intervention_module(intervention_module):
    for parameter in intervention_module.parameters():
        parameter.requires_grad = False
    return intervention_module


def get_next_block(model, intervention_layer, low_rank_dimension, kwargs):

    next_block_idx = intervention_layer + 1  # the block to act is the one that receives corrupted interventions
    originial_next_block = model.model.layers[next_block_idx]
    custom_next_block = LlamaBlockDefendor(originial_next_block, low_rank_dimension, kwargs)
    return custom_next_block


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


def get_red_teaming_data_module(model, tokenizer, attack_data_dict, process_config, eval=False):
    """
    #TODO implement suffling?
    """
    if eval:
        samples_to_use = process_config['safety_eval_dataset_size']
    else:
        samples_to_use = process_config['dataset_size']
    # Assuming full position here, TODO implement partial positions...
    intervetion_locations = [[[[-1]] * process_config['intervention_count']]] * samples_to_use
    intervetion_locations = torch.Tensor(intervetion_locations).squeeze(1)
    if eval:
        train_dataset = datasets.Dataset.from_dict({
            "input_ids": attack_data_dict['all_base_input_ids'][:samples_to_use],
            "intervention_locations": intervetion_locations
        })
    else:
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
        logging_dict,
        kwargs):
    report_attack_config(attack_config, logging_dict, kwargs)
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
    """
    safety_eval_data_module = get_red_teaming_data_module(
        model, tokenizer, safety_eval_data, attack_config, True)

    safety_eval_dataloader = DataLoader(
            safety_eval_data_module['train_dataset'],
            batch_size=attack_config['batch_size'],
            collate_fn=safety_eval_data_module['data_collator']
        )
    
    eval_safety(reft_model, eval_model, eval_tokenizer, safety_eval_dataloader, kwargs['device'])
    """
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

    logging_dict['wandb_run'].log(
        { TOXICITY_AFTER_ATTACK: attack_config['toxicity'],
         STEP_LABEL: kwargs['timestep'] })

    return reft_model, eval_table


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

    with torch.no_grad():

        flag = 0
        eval_step = 0
        eval_table = wandb.Table(columns=['prompt', 'response', 'moderation'])
        for eval_prompt, tokenized_eval_prompt in zip(
                safety_eval_data['red_team_prompts_no_template'][:eval_prompts], 
                safety_eval_data['all_base_input_ids'][:eval_prompts]):
            
            # if kwargs['verbose']: print('request: ',eval_prompt)
            output = deterministic_query(
                tokenized_eval_prompt,
                model,
                model_tokenizer,
                isReftModel,
                kwargs['device']
            )
            # if kwargs['verbose']: print('output: ',output)
            data_tuple = [{"role": "user",
                            "content": eval_prompt},
                            {"role": "assistant",
                            "content": output}]
        
            moderation = moderate(data_tuple, eval_model, eval_tokenizer, kwargs['device'])
            unsafe_cat = 'safe'
            if moderation != 'safe':
                flag += 1
                unsafe_cat = moderation.split('\n')[-1]
            # if kwargs['verbose']: print('moderation: ',unsafe_cat)
            eval_table.add_data(eval_prompt, output, unsafe_cat)
            eval_step += 1
        
        toxicity_score = flag/eval_step

        if mode == 'init':
            logging_dict['wandb_run'].log({"Init_behaviour" : eval_table})

        '''
        if mode == 'attack':
            logging_dict['wandb_run'].log(
                {"After Attack Behaviour. Layer: " \
                    + str(kwargs['current_layer']) + " Step: " \
                    + str(kwargs['timestep']) : eval_table})
        elif mode == 'defence': 
            logging_dict['wandb_run'].log(
                {"After Defence Behaviour. Layer: " \
                    + str(kwargs['current_layer']) + " Step: " \
                    + str(kwargs['timestep']): eval_table})
        '''
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
        tokenized_prompt, model, tokenizer, isReftModel, device):

    tokenized_prompt = tokenized_prompt.to(device)
    input_len = len(tokenized_prompt)
        
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
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            )
    else:
            s = model.generate(
                inputs=tokenized_prompt.unsqueeze(0),
                top_p=1,
                temperature=1.0,  # greedy decoding
                do_sample=False,  # greedy decoding
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
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


def defence_training_loop(
    defence_config, 
    defence_dataloader, 
    intervenable_model, 
    defence_criterion, 
    defence_optimizers, 
    kwargs):

    mean_loss = 0
    mean_defensive_loss = 0
    mean_reg_loss = 0
    defence_optimizer = defence_optimizers[0]
    if kwargs['defence_regularization'] == 'compound':
        reg_optimizer = defence_optimizers[1]

    if kwargs['tqdm']: ranger = trange
    else: ranger = range

    for epoch in ranger(defence_config['epochs']):

        epoch_defensive_loss = 0
        epoch_reg_loss = 0
        epoch_loss = 0

        for batch_idx, batch in enumerate(defence_dataloader):
            
            with torch.no_grad():

                intervention_locations = batch['intervention_locations'].permute(1, 0, 2).tolist()
                position_ids = torch.tensor(intervention_locations[0][0], device=kwargs['device']).unsqueeze(0)
                batch.to(kwargs['device'])
                intervention_outputs = intervenable_model(
                    base={'input_ids' : batch['input_ids'],
                        'attention_mask': batch['attention_mask']},
                    unit_locations={"base" : intervention_locations})
                
                ### We assume for now only one-block defence and one-block attacks.  # TODO adjust indexing here for multi-block defences
                original_input_representations = torch.vstack([output.unsqueeze(0) for output in intervention_outputs[0][1]])     
                # for defence_layer, defence_modules  in defence_config['defences'].items():  # TODO iterate over multiple defences 
                defence_layer, defence_module = list(defence_config['defences'].items())[0]

                corruption_module = defence_module['adversarial_intervention_module']
                corrupted_input_reps = corruption_module(original_input_representations)  # these are our "inputs"
                defensive_block = defence_module['next_transformer_block']
                original_output_reps = defensive_block(
                    original_input_representations, 
                    position_ids=position_ids)[0]  # these are our "labels"


            regularizing_output_reps = defensive_block.intervened_forward(
                original_input_representations,
                position_ids=position_ids)[0]

            regularization_term = defence_criterion(
                input=regularizing_output_reps,
                target=original_output_reps)

            if kwargs['defence_regularization'] == 'compound':
                reg_optimizer.zero_grad()
                regularization_term.backward()
                reg_optimizer.step()
            
            epoch_reg_loss += regularization_term.detach()    
            corrupted_output_reps = defensive_block.intervened_forward(
                corrupted_input_reps,
                position_ids=position_ids)[0]

            defensive_loss = defence_criterion(
                input=corrupted_output_reps,
                target=original_output_reps)

            if kwargs['defence_regularization'] == 'simple':
                main_loss = defensive_loss + regularization_term
            else:
                main_loss = defensive_loss

            defence_optimizer.zero_grad()
            main_loss.backward()
            defence_optimizer.step()
            epoch_defensive_loss += defensive_loss.detach()        
            epoch_loss += main_loss.detach()

        epoch_reg_loss /= batch_idx
        epoch_defensive_loss /= batch_idx
        epoch_loss /= batch_idx

        """ 
        if kwargs['verbose'] and not kwargs['tqdm']: 
            print(f'defence epoch {epoch} mean defensive loss {epoch_defensive_loss}')
            print(f'defence epoch {epoch} mean regularization loss {epoch_reg_loss}')
            print(f'defence epoch {epoch} mean loss {mean_epoch_loss}')
        """
        mean_defensive_loss += epoch_defensive_loss.item()
        mean_reg_loss += epoch_reg_loss.item()
        mean_loss += epoch_loss.item()

    mean_reg_loss /= defence_config['epochs']
    mean_defensive_loss /= defence_config['epochs']
    mean_loss /= defence_config['epochs']

    defence_results = {
        'mean_reg_loss':  mean_reg_loss,
        'mean_defensive_loss': mean_defensive_loss,
        'mean_loss' : mean_loss
    }

    return defence_results  


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
            { DEFENCE_REG_LOSS: defence_results['mean_loss'],
            STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { DEFENCE_DEF_LOSS: defence_results['mean_loss'],
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
        kwargs)

    model = reset_defended_module(model, defence_config, kwargs)

    logging_dict['wandb_run'].log(
        { TOXICITY_AFTER_DEFENCE: toxicity_score,
         STEP_LABEL: kwargs['timestep'] })
    defence_config['safety'] = 1 - toxicity_score
    logging_dict['wandb_run'].log(
        { SAFETY_AFTER_DEFENCE: defence_config['safety'],
         STEP_LABEL: kwargs['timestep'] })
    logging_dict['wandb_run'].log(
        { PERFORMANCE_AFTER_DEFENCE: defence_config['performance'],
         STEP_LABEL: kwargs['timestep'] })

    return eval_table

    
def get_defence_optimizers(defence_config, kwargs):
    parameters_for_defence_optimizer = []
    parameters_for_regularization_optimizer = []
    if kwargs['defence_regularization'] == 'simple':
        for adv_intervention_layer, defence_modules_dict in defence_config['defences'].items():
            defence_block = defence_modules_dict['next_transformer_block']
            # all loras are used for defence + reg. (could lead to gradient conflict)
            parameters_for_defence_optimizer.extend([param for param in defence_block.parameters() if param.requires_grad])
        return [torch.optim.Adam(parameters_for_defence_optimizer, lr=kwargs['learning_rate'])]
    else:
        assert kwargs['defence_regularization'] == 'compound' and \
            'GATE' in kwargs['defence_strategy'] and \
            'UP' in kwargs['defence_strategy'] and \
            'DOWN' in kwargs['defence_strategy']
        for adv_intervention_layer, defence_modules_dict in defence_config['defences'].items():
            defence_block = defence_modules_dict['next_transformer_block']
            # gate and up projections are used for defence:
            for named_param in defence_block.named_parameters():
                if named_param[0] in ['gate_A.weight', 'gate_B.weight', 'up_A.weight', 'up_B.weight']:\
                    parameters_for_defence_optimizer.append(named_param[1])
                elif named_param[0] in ['down_A.weight', 'down_B.weight']:
                    parameters_for_regularization_optimizer.append(named_param[1])
        return [torch.optim.Adam(parameters_for_defence_optimizer, lr=kwargs['learning_rate']),
                torch.optim.Adam(parameters_for_regularization_optimizer, lr=kwargs['learning_rate'])]

def get_defence_dataloader(model, tokenizer, defence_config, attack_data_dict):
    data_module = get_red_teaming_data_module(
        model, tokenizer, attack_data_dict, defence_config)

    return DataLoader(
            data_module['train_dataset'],
            batch_size=defence_config['batch_size'],
            collate_fn=data_module['data_collator']
        )


def absorb_defender_adaptor(model, defence_config, kwargs):
    
    kwargs['cached_original_modules'] = {'UP': {},
                                         'GATE': {},
                                         'DOWN': {}}

    for defence_layer, defence_module in defence_config['defences'].items():
        if kwargs['verbose']: print(f'Absorbing defence LoRA in decoder layer {defence_layer+1}...')
        defensive_block = defence_module['next_transformer_block']

        if 'GATE' in kwargs['defence_strategy']:
            kwargs['cached_original_modules']['GATE'][defence_layer+1] = model.model.layers[defence_layer+1].mlp.gate_proj.weight.clone()
            defensive_lora_adaptor = torch.matmul(
                defensive_block.mlp.gate_B.weight, 
                defensive_block.mlp.gate_A.weight)
            model.model.layers[defence_layer+1].mlp.gate_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer+1].mlp.gate_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        if 'UP' in kwargs['defence_strategy']:
            kwargs['cached_original_modules']['UP'][defence_layer+1] = model.model.layers[defence_layer+1].mlp.up_proj.weight.clone()
            defensive_lora_adaptor = torch.matmul(
                defensive_block.mlp.up_B.weight, 
                defensive_block.mlp.up_A.weight)
            model.model.layers[defence_layer+1].mlp.up_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer+1].mlp.up_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

        if 'DOWN' in kwargs['defence_strategy']:
            kwargs['cached_original_modules']['DOWN'][defence_layer+1] = model.model.layers[defence_layer+1].mlp.down_proj.weight.clone()
            defensive_lora_adaptor = torch.matmul(
                defensive_block.mlp.down_B.weight, 
                defensive_block.mlp.down_A.weight)
            model.model.layers[defence_layer+1].mlp.down_proj.weight = torch.nn.Parameter(
                model.model.layers[defence_layer+1].mlp.down_proj.weight.clone() + 
                (defence_config['absortion_scaling'] * defensive_lora_adaptor))

    return model


def reset_defended_module(model, defence_config, kwargs):
    
    for defence_layer, defence_module in defence_config['defences'].items():
        if kwargs['verbose']: print(f'Resetting defence LoRA defences in layer {defence_layer+1}...')
        if 'GATE' in kwargs['defence_strategy']:
            model.model.layers[defence_layer+1].mlp.gate_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['GATE'][defence_layer+1])
        if 'UP' in kwargs['defence_strategy']:
            model.model.layers[defence_layer+1].mlp.up_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['UP'][defence_layer+1])
        if 'DOWN' in kwargs['defence_strategy']:
            model.model.layers[defence_layer+1].mlp.down_proj.weight = torch.nn.Parameter(kwargs['cached_original_modules']['DOWN'][defence_layer+1])

    return model


def evolve_attack_config(model, layer, prev_attack_config, kwargs):
    """
    TODO implement
    """
    attack_config = init_single_layer_attack_config(model, layer, kwargs)
    attack_config['dataset_size'] = prev_attack_config['dataset_size'] + 50 
    if attack_config['dataset_size'] > kwargs['max_red_teaming_dataset_size']:
        attack_config['dataset_size'] = kwargs['max_red_teaming_dataset_size']
        if attack_config['epochs'] < 20:
            attack_config['epochs'] += 2
    return attack_config


def evolve_defence_config(model, attack_config, attacked_model, prev_defence_config, kwargs):
    defence_config = init_custom_defence_config(model, attack_config, attacked_model, kwargs)
    defence_config['dataset_size'] = prev_defence_config['dataset_size'] + 50 
    if defence_config['dataset_size'] > kwargs['max_red_teaming_dataset_size']:
        defence_config['dataset_size'] = kwargs['max_red_teaming_dataset_size']
        if defence_config['epochs'] < 20:
            defence_config['epochs'] += 2
    if defence_config['regularization_coefficient'] < 2:
        defence_config['regularization_coefficient'] += 0.3
    return defence_config


def pprint_attack_config(attack_config):
    # TODO implement selective reporting...
    print('Dataset size: ', attack_config['dataset_size'])
    

def final_report(logging_dict):
        print('IMMUNIZATION REPORT:\n\n')
        pprint(logging_dict['immunization_report'])
        logging_dict['wandb_run'].log({'IMMUNIZATION REPORT':logging_dict['immunization_table']})
