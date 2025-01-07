#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python eval.py tvac

CUDA_VISIBLE_DEVICES=0 python eval.py vac

CUDA_VISIBLE_DEVICES=0 python eval.py repnoise

CUDA_VISIBLE_DEVICES=0 python eval.py booster




