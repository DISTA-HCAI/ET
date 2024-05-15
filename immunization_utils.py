import transformers
import torch
import sys
sys.path.append('/home/jovyan/pyreft/pyvene')
sys.path.append('/home/jovyan/pyreft')
import pyreft
from pyreft.interventions import \
        NoreftIntervention, \
        LoreftIntervention, \
        LoreftInterventionNoBias, \
        LobireftIntervention, \
        NodireftIntervention
import peft
from peft import (  
    LoraConfig, 
    LoftQConfig, 
    LoHaConfig, 
    LoftQConfig)

INTERVENTIONS = [
    NoreftIntervention,
    LoreftIntervention,
    LoreftInterventionNoBias,
    LobireftIntervention,
    NodireftIntervention]

LORAS = [LoraConfig, 
        LoftQConfig, 
        LoHaConfig, 
        LoftQConfig]

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


def lora_defense(model, reft_intervention, lora_hyper_params):
    reft_model = get_reft_model(model, reft_intervention)
    lora_adaptor = get_lora_adaptor(lora_hyper_params)
    return train_and_eval_lora(model, lora_adaptor, reft_model)


def init_attack_config(model, kwargs):

    representations = []

    for layer in kwargs['init_attack_layers']:

        for intervention_place in kwargs['intervention_places'].split(';'):

            if intervention_place not in ['mlp_gate_output', 'mlp_up_output']:
                embed_dim = model.config.hidden_size  
            else: embed_dim = model.config.intermediate_size

            # Step 3: Retrieve the intervention name:
            InterventionClass = globals()[kwargs['init_intervention_type']]

            representations.append({
                    "layer": layer,
                    "component": intervention_place,
                    "low_rank_dimension": kwargs['init_low_rank_attack_dimension'],
                    "intervention": InterventionClass(
                                        embed_dim=embed_dim, 
                                        low_rank_dimension=kwargs['init_low_rank_attack_dimension'])
                })

    return pyreft.ReftConfig(representations=representations)


def init_defense_config(model, kwargs):

    LoraClass = globals()[kwargs['init_defence_class']]


    # Define the LoRA configuration
    lora_config = LoraClass(
        use_dora=kwargs['init_use_dora'],
        r=kwargs['init_low_rank_defense_dimension'],
        lora_alpha=kwargs['init_defence_scaling_factor'],  # Scaling factor
        lora_dropout=kwargs['init_defence_dropout'],  # Dropout probability
        target_modules=kwargs['init_defence_target_modules'].split(';'),  # Target the gate and up projections in MLP blocks
    )
    return lora_config