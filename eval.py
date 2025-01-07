from transformers import AutoModelForCausalLM
import lm_eval
import os
from peft import PeftModel
import json
import sys
from lm_eval.loggers import WandbLogger
import wandb

# read the lora adaptor param from command line:
# python eval.py tvac

if len(sys.argv) < 2:
    print('Please provide the baseline path as a command line argument.')
    sys.exit(1)

baseline_name = sys.argv[1]

all_tiny_tasks = [
    "tinyArc",
    "tinyGSM8k",
    "tinyHellaswag",
    "tinyMMLU",
    "tinyTruthfulQA",
    "tinyTruthfulQA_mc1",
    "tinyWinogrande",
]

lora_adaptors = {'tvac': '/home/jovyan/.cache/huggingface/hub/t_vaccine/Meta-Llama-3-8B-Instruct_TVaccine_3_8_20_200_200_20',
                'vac': '/home/jovyan/.cache/huggingface/hub/vaccine/Meta-Llama-3-8B-Instruct_vaccine_3_20',
                'booster': '/home/jovyan/.cache/huggingface/hub/booster/Meta-Llama-3-8B-Instruct_smooth_5_0.1_5000_5000',
                'tar': '/home/jovyan/.cache/huggingface/hub/tar/Meta-Llama-3-8B-Instruct_tar_20',
                'repnoise': '/home/jovyan/.cache/huggingface/hub/repnoise/Meta-Llama-3-8B-Instruct_repnoise_0.1_0.001_20',
                'no-imm': None,
                'cb': None,
                'debug': None,
                'et':None
                }

base_model = {'tvac': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'vac': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'booster': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'tar': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'repnoise': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'no-imm': 'meta-llama/Meta-Llama-3-8B-Instruct',
                'cb': '/home/jovyan/.cache/huggingface/hub/Llama-3-8b_CB',
                'debug': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'et': '/home/jovyan/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct_ET'
                }

print(f'Loading the model {base_model[baseline_name]} that we want to eval...\n\n')

model = AutoModelForCausalLM.from_pretrained(
    base_model[baseline_name],
    cache_dir='/home/jovyan/.cache/huggingface/hub',
    device_map='cuda'
    )

if lora_adaptors[baseline_name] is not None:
    print(f'Loading the lora adaptor...{lora_adaptors[baseline_name]}\n\n')
    model = PeftModel.from_pretrained(
                model,
                lora_adaptors[baseline_name],
                )

    print('Merging the lora adaptor...\n\n')
    model = model.merge_and_unload()

print('Initializing lm eval...\n\n')
model = lm_eval.models.huggingface.HFLM(pretrained=model)

task_manager = lm_eval.tasks.TaskManager()

print('will evaluate now...\n\n')
results= {}

wandb_logger = WandbLogger(
    project="low_cost_toxification", 
    job_type="eval",
    name="eval_"+baseline_name,
    tags = ["NEW_LOGGING", "DEFENCE_FIX"]
)

results = lm_eval.simple_evaluate(
    model=model,  # Pass your loaded model object here
    tasks=all_tiny_tasks,
    task_manager=task_manager,
    device='cuda',
    log_samples=True,
)

wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results["samples"])  # if log_samples

wandb.finish()