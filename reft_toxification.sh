#!/bin/bash

# Define the list of layers
LAYERS_LIST=("18;28" )

for LAYER in "${LAYERS_LIST[@]}"
do
  # Command to run
  python3 reft.py \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
    --layers=$LAYER \
    --low_rank_dimension=2 \
    --n_train_examples=100 \
    --batch_size=10 \
    --learning_rate=4e-3 \
    --num_train_epochs=5 \
    --output_dir=local_checkpoints \
    --logging_steps=1 \
    --positions=all \
    --share_weights \
    --intervention_places=block_output \
    --nonstop \
    --mode=toxify \
    --data_path=data/harmful_behaviors.csv \
    --logging

done


