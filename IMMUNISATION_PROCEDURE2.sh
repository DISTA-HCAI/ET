#!/bin/bash

    

: << 'EXPERIMENT_1_MLP_ATTACK_AND_DEFENCE'
CUDA_VISIBLE_DEVICES=0 python3 immunization.py \
    --verbose --logging --torch_seed=3 --save \
    --min_performance_percentage_defence=0.5 \
    --min_performance_percentage_attack=0.2 \
    --init_attack_intervention_places=mlp \
    --run_name=experimental_vaccine --init_low_rank_defence_dimension=2
EXPERIMENT_1_MLP_ATTACK_AND_DEFENCE

: << 'EXPERIMENT_2_BLOCK_ATTACK_AND_DEFENCE'
CUDA_VISIBLE_DEVICES=0 python3 immunization.py \
    --verbose --logging --torch_seed=3 --save \
    --min_performance_percentage_defence=0.5 \
    --min_performance_percentage_attack=0.2 \
    --init_attack_intervention_places=block \
    --run_name=new_vaccine_5_replication --init_low_rank_defence_dimension=2
EXPERIMENT_2_BLOCK_ATTACK_AND_DEFENCE


CUDA_VISIBLE_DEVICES=1 python3 immunization.py \
    --verbose --logging --torch_seed=3 \
    --min_performance_percentage_defence=0.7 \
    --min_performance_percentage_attack=0.2 \
    --init_attack_intervention_places=block \
    --multiblock_defences \
    --run_name=multi_block --init_low_rank_defence_dimension=2

CUDA_VISIBLE_DEVICES=1 python3 immunization.py \
    --verbose --logging --torch_seed=3 \
    --min_performance_percentage_defence=0.7 \
    --min_performance_percentage_attack=0.2 \
    --init_attack_intervention_places=block \
    --run_name=single_block --init_low_rank_defence_dimension=2