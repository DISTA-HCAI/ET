import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import tqdm
from pprint import pprint
from immunization_utils import *
from tqdm import tqdm, trange


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):

    if cfg.override != "":
        try:
            # Load the variant specified from the command line
            config_overrides = OmegaConf.load(hydra.utils.get_original_cwd() + f'/config/overrides/{cfg.override}.yaml')
            # Merge configurations, with the variant overriding the base config
            cfg = OmegaConf.merge(cfg, config_overrides)
        except:
            print('Unsuccesfully tried to use the configuration override: ',cfg.override)

    kwargs, \
        logging_dict, \
        model, tokenizer, \
        eval_model, eval_tokenizer, \
        training_attack_data_dict, \
        safety_eval_data, \
        performance_eval_data = initialize(cfg)
    
    post_successful_attack_behaviour = None
    post_failed_attack_behaviour = None

    #
    # Immunization loop:
    #

    if kwargs['reverse']:
        ranger = range(kwargs['starting_layer'], -1, -1)
    else:
        ranger = range(kwargs['starting_layer'], model.config.num_hidden_layers)
        
    for layer in ranger:

        # Reporting qualitative results of a successful immunization on the previous layer (for the paper):
        if (post_successful_attack_behaviour is not None and post_failed_attack_behaviour is not None) : 
            report_qualitative_immunisation_results(
                post_successful_attack_behaviour, 
                post_failed_attack_behaviour, 
                logging_dict, kwargs)
        
        #
        # The very layer-wise immunization process starts here:
        #
        
        if kwargs['verbose']: print(f'Immunizing layer {layer} \n')
        kwargs['timestep'] += 1
        logging_dict['wandb_run'].log({'IMMUNIZING_LAYER': layer, 'STEP': kwargs['timestep']})

        outer_attack_rounds = outer_defence_rounds = 0
        defence_config = attack_config = None
        kwargs['current_layer'] = layer
        layer_immunized = defence_succeeded = False
        post_successful_attack_behaviour = post_failed_attack_behaviour = None
        
        # The following end condition prevents eternal war between attack/defense rounds...
        for immunization_round in range(kwargs['max_immunization_rounds']):  

            if kwargs['verbose']: print('Attacking phase...\n')
            attack_succeeded = False
            outer_attack_rounds += 1

            # Initialize a configuration for attack
            attack_config = init_single_layer_attack_config(model, layer, kwargs)  
            
            # We have a maximum number of attack attempts:
            for inner_attack_round in range(kwargs['max_attack_rounds']):
                
                kwargs['timestep'] += 1
                
                attacked_model, safety_eval_table = reft_attack(
                    model, 
                    tokenizer, 
                    attack_config, 
                    training_attack_data_dict, 
                    eval_model, 
                    eval_tokenizer, 
                    safety_eval_data,
                    performance_eval_data,
                    logging_dict, 
                    kwargs)

                log_step(layer, 'attack', attack_config['toxicity'], attack_config['performance'], logging_dict, kwargs)
                
                # Did we make an efficacious attack?
                if is_successful_attack(attack_config, kwargs):
                      
                    if kwargs['verbose']: print('Attack succeeded! Absolute Toxicity: ', attack_config['toxicity'], ' Absolute Performance: ', attack_config['performance'],'\n')
                    log_successful_step(layer, 'attack', attack_config['toxicity'], attack_config['performance'], logging_dict, kwargs)
                    attack_succeeded = True
                    post_successful_attack_behaviour = safety_eval_table
                    if not kwargs['baseline']:
                        break  # end attack round

                # attack failed, try a new attack configuration
                else:  
                    if kwargs['verbose']: print('Attack failed! Absolute Toxicity: ', attack_config['toxicity'], ' Absolute Performance: ', attack_config['performance'],'\n')
                    # Some failed attacks are however quite toxic, we wont show that on the qualitative results:
                    if attack_config['toxicity'] <= 0.5: post_failed_attack_behaviour = safety_eval_table
                    # Can we try again the attack?
                    if inner_attack_round < kwargs['max_attack_rounds'] - 1:
                        if kwargs['verbose']: print(' Evolving attack...\n')
                        attack_config = evolve_attack_config(model, layer, attack_config, kwargs)

            # Did we fail to craft an efficacious attack given our training budget?
            if not attack_succeeded: 
                immunized = True
                print(f'Layer {layer} got immunized after {outer_attack_rounds} attack and {outer_defence_rounds} defence rounds!')
                log_immunization(
                    layer, 
                    True, 
                    outer_attack_rounds,
                    outer_defence_rounds,
                    attack_config, 
                    defence_config,
                    logging_dict,
                    kwargs)
                layer_immunized = True
                break  # go to next layer

            # If the attack was successful instead, we start the defensive phase. 
            else: 

                if kwargs['verbose']: print('Defensive phase...\n')
                outer_defence_rounds += 1
                
                # Initialize a configuration for defence:
                defence_config = init_custom_defence_config(model, attack_config, attacked_model, 1, kwargs)
                # If defence rounds are adding blocks of intervention each time, then they might be less than the fixed
                # max_defense rounds...
                max_defence_rounds = get_max_defence_rounds(model, layer, kwargs)
                defence_succeeded = False
                                
                for inner_defence_round in range(max_defence_rounds):
                    kwargs['timestep'] += 1

                    # train a defensive module!
                    custom_defence(
                        model,
                        tokenizer,
                        eval_model,
                        eval_tokenizer,
                        defence_config,
                        training_attack_data_dict,
                        safety_eval_data,
                        performance_eval_data,
                        logging_dict, 
                        kwargs)
                    log_step(layer, 'defence', 1 - defence_config['safety'], defence_config['performance'], logging_dict, kwargs)
                    
                    # Did we make an efficacious defence?
                    if is_successful_defence(defence_config, kwargs): 
                        if kwargs['verbose']: print('Defence succeded! Relative Safety: ', defence_config['safety'] ,' Absolute Performance :', defence_config['performance'] ,'\n')
                        
                        # Are we absorbing defences on the go?
                        if not kwargs['memory_less']:
                            model = absorb_defender_adaptor(model, defence_config, kwargs)
                            logging_dict['wandb_run'].log({'Absorbed defences at layer': layer, STEP_LABEL: kwargs['timestep']})
                        
                        # Shall we save our defensive modules?
                        if kwargs['save']:
                            save_model(model, layer, kwargs)

                        log_successful_step(layer, 'defence', 1 - defence_config['safety'], defence_config['performance'], logging_dict, kwargs)
                        defence_succeeded = True
                        break  # end defence round

                    # defence failed, try a new defence configuration if possible
                    else:  
                        if kwargs['verbose']: print('Defence failed! Relative Safety: ', defence_config['safety'] ,' Absolute Performance :', defence_config['performance'])
                        if inner_defence_round < max_defence_rounds - 1:
                            if kwargs['verbose']: print('Evolving defence... \n')
                            defence_config = evolve_defence_config(model, attack_config, attacked_model, defence_config, kwargs)
                
                # Did we fail to defent against this attack given a defence budget?
                if not defence_succeeded:  
                    print(f'Failed to find a defence for last attack at layer {layer}')
                    # pprint_attack_config(attack_config)
                    log_immunization(
                        layer, 
                        False, 
                        outer_attack_rounds,
                        outer_defence_rounds,
                        attack_config, 
                        defence_config,
                        logging_dict,
                        kwargs)
                    if kwargs['verbose']: print('Will continue with next layers... \n')
                    break  # go to next layer


        if defence_succeeded and (immunization_round == kwargs['max_immunization_rounds'] - 1):   
            # we defended our model on each attack, and no more immunization rounds available...
            print(f'Tie at layer {layer}')
            log_immunization(
                        layer, 
                        False, 
                        outer_attack_rounds,
                        outer_defence_rounds,
                        attack_config, 
                        defence_config,
                        logging_dict,
                        kwargs)

    final_report(logging_dict)

    

if __name__ == "__main__":

    main()