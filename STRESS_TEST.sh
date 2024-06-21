#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 immunization.py \
    --verbose --logging --torch_seed=3 \
    --max_defence_rounds=0 --max_attack_rounds=4 --max_immunization_rounds=5 \
    --init_attack_prompts=500 --baseline --mount_vaccines=VACCINE_COMPOUND_reg20s* \
    --vaccine_weight=0.8 --run_name=TESTING_compound_reg20