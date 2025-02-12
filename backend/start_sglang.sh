#!/bin/bash

model=meta-llama/Meta-Llama-3.1-8B-Instruct
model=lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit
model=allenai/Llama-3.1-Tulu-3-8B-DPO
model_path=${1:-$model}

python -m sglang.launch_server --model-path $model_path --port 30000 --host 0.0.0.0 
