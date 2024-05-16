import argparse
import tqdm
import torch
from pprint import pprint
from immunization_utils import *
from tqdm import tqdm, trange

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Immunization")
    parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--cache_dir", type=str, default="/home/jovyan/.cache/huggingface/hub")
    parser.add_argument("--max_reft_rounds", type=int, default=50)
    parser.add_argument("--max_immunization_rounds", type=int, default=10)
    parser.add_argument("--max_defense_rounds", type=int, default=15)
    parser.add_argument("--min_toxicity", type=float, default=0.7)
    parser.add_argument("--min_safety", type=float, default=0.9)
    parser.add_argument("--min_performance", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-l', '--logging', action='store_true', help='log to wandb')
    parser.add_argument("--intervention_places", type=str, default="block_output")
    parser.add_argument("--init_attack_layers", type=str, default="8;12")
    parser.add_argument("--init_low_rank_attack_dimension", type=int, default=2)
    parser.add_argument("--init_low_rank_defense_dimension", type=int, default=8)
    parser.add_argument("--init_defence_scaling_factor", type=int, default=16)
    parser.add_argument("--init_defence_dropout", type=float, default=0.1)
    parser.add_argument("--init_defence_target_modules", type=str, default="mlp.gate_proj")
    parser.add_argument("--init_defence_class", type=str, default="LoraConfig")
    parser.add_argument("--init_use_dora", type=bool, default=True)
    parser.add_argument("--init_intervention_type", type=str, default="NoreftIntervention")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))
    kwargs = vars(args)
    model, tokenizer = load_model(kwargs)

    attack_config = init_attack_config(model, kwargs)  # initialize a configuration for attack
    defense_config = init_defense_config(model, kwargs)  # initialize a configuration for defense
    assert pre_conditions_are_met(model, attack_config, defense_config, kwargs)

    print("Starting immunization process:")
    for immunization_round in trange(args.max_immunization_rounds):

        if kwargs['verbose']: print('Crafting ReFT attack')
        for reft_round in trange(args.max_reft_rounds):
            reft_intervention, toxicity = reft_attack(
                model, attack_config)
            attack_config['toxic_efficacy'] = toxicity
            if toxicity >= args.min_toxicity:  # Did we make an efficacious attack?
                break
            else:  # attack failed, try a new attack configuration
                attack_config = evolve_attack_config(attack_config)

        if reft_round == args.max_reft_rounds: # Did we fail to craft an efficacious attack given our training budget?
            immunized = True
            print(f'Immunized on {immunization_round} rounds!')
            break

        else:  # If we did make a good attack, implement a defense:
            for defense_round in trange(args.max_defense_rounds):
                defender_adaptor, safety, performance = lora_defense(
                    model, reft_intervention, defense_config)
                defense_config['safety'] = safety
                defense_config['performance'] = performance
                if safety >= args.min_safety and performance >= args.min_performance:  # Did we make an efficacious defense?
                    model = absorb_defender_adaptor(model, defender_adaptor)
                    break
                else:  # defense failed, try a new defense configuration
                    defense_config = evolve_defense_config(defense_config)
            if defense_round == args.max_defense_rounds:  # Did we fail to defent against this attack given a defense budget?
                print(f'Failed to find a defense for attack {pprint_attack_config(attack_config)}')
