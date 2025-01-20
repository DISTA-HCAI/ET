#!/bin/bash

# Create an immused model:
CUDA_VISIBLE_DEVICES=0 python3 immunization.py run_name=et

# Benchmark it's toxicity against ITI-attacks:
CUDA_VISIBLE_DEVICES=0 python3 immunization.py override=benchmark run_name=et_benchmark mount_vaccines=et*

# Evaluate the performance also: (check config for adjusting saving paths...)
CUDA_VISIBLE_DEVICES=0 python3 eval.py et


