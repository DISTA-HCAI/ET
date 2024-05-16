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
    parser.add_argument("--max_defence_rounds", type=int, default=15)
    parser.add_argument("--min_toxicity", type=float, default=0.5)
    parser.add_argument("--min_safety", type=float, default=0.9)
    parser.add_argument("--min_performance", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-l', '--logging', action='store_true', help='log to wandb')
    parser.add_argument("--red_teaming_data_path", type=str, default="data/harmful_behaviors.csv")
    parser.add_argument("--red_teaming_input_col", type=str, default="goal")
    parser.add_argument("--red_teaming_label_col", type=str, default="target")
    parser.add_argument("--learning_rate", type=float, default=4e-3)
    
    parser.add_argument("--template", type=str, default="assistant")

    parser.add_argument("--init_attack_batch_size", type=int, default=10)
    parser.add_argument("--init_defence_batch_size", type=int, default=10)

    parser.add_argument("--init_attack_intervention_places", type=str, default="block_output")
    parser.add_argument("--init_defence_intervention_places", type=str, default="mlp_up_output")

    parser.add_argument("--init_attack_positions", type=str, default="all")  # TODO use these...
    parser.add_argument("--init_defence_positions", type=str, default="all") # TODO does it have sense to play with defence positions?...

    parser.add_argument("--init_attack_layers", type=str, default="8;12")
    parser.add_argument("--init_defence_layers", type=str, default="8;12")

    parser.add_argument("--init_low_rank_attack_dimension", type=int, default=2)
    parser.add_argument("--init_low_rank_defence_dimension", type=int, default=8)

    parser.add_argument("--init_attack_dropout", type=float, default=0.1)
    parser.add_argument("--init_defence_dropout", type=float, default=0.1)
    
    parser.add_argument("--init_attack_intervention_type", type=str, default="NoreftInterventionNoBias")
    parser.add_argument("--init_defence_intervention_type", type=str, default="NoreftInterventionNoBias")

    parser.add_argument("--init_attack_prompts", type=int, default="50")
    parser.add_argument("--init_defence_prompts", type=int, default="50")

    parser.add_argument("--init_attack_epochs", type=int, default="5")
    parser.add_argument("--init_defence_epochs", type=int, default="5")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))
    kwargs = vars(args)
    model, tokenizer = load_model(kwargs)

    attack_data_dict = load_red_teaming_data(tokenizer, kwargs)
    

    attack_config = init_attack_config(model, kwargs)  # initialize a configuration for attack
    assert pre_conditions_are_met(model, attack_config, kwargs)

    print("Starting immunization process:")
    for immunization_round in trange(args.max_immunization_rounds):

        if kwargs['verbose']: print('Crafting ReFT attack')
        for reft_round in trange(args.max_reft_rounds):
            attacked_model, attack_results = reft_attack(
                model, tokenizer, attack_config, attack_data_dict, kwargs)
            attack_config['toxicity'] = get_toxicity(attack_results)
            if attack_config['toxicity'] >= args.min_toxicity:  # Did we make an efficacious attack?
                print('Attack succeeded')
                break
            else:  # attack failed, try a new attack configuration
                print('Attack failed')
                attack_config = evolve_attack_config(attack_config)

        if reft_round == args.max_reft_rounds: # Did we fail to craft an efficacious attack given our training budget?
            immunized = True
            print(f'Immunized on {immunization_round} rounds!')
            break

        else:  # If we did make a good attack, implement a defence:
            defence_config = init_custom_defence_config(model, attack_config, attacked_model, kwargs)
            for defence_round in trange(args.max_defence_rounds):
                defender_adaptor, safety, performance = reft_defence(
                    model, tokenizer, reft_interventions, defence_config)
                defence_config['safety'] = safety
                defence_config['performance'] = performance
                if safety >= args.min_safety and performance >= args.min_performance:  # Did we make an efficacious defence?
                    model = absorb_defender_adaptor(model, defender_adaptor)
                    break
                else:  # defence failed, try a new defence configuration
                    defence_config = evolve_defence_config(defence_config)
            if defence_round == args.max_defence_rounds:  # Did we fail to defent against this attack given a defence budget?
                print(f'Failed to find a defence for attack {pprint_attack_config(attack_config)}')
