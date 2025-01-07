#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval.py tar

CUDA_VISIBLE_DEVICES=0 python eval.py no-imm

CUDA_VISIBLE_DEVICES=0 python eval.py cb





