#!/bin/bash

    
CUDA_VISIBLE_DEVICES=1 python3 immunization.py \
    --verbose --logging --torch_seed=3 --save \
    --min_performance_percentage_defence=0.5 \
    --min_performance_percentage_attack=0.2 \
    --init_attack_intervention_places=mlp_output \
    --run_name=NEW_VACCINE_5 --init_low_rank_defence_dimension=2