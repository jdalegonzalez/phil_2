#!/bin/bash

#model="meta-llama/Meta-Llama-3.1-8B-Instruct"
#model="lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit"
#model="allenai/Llama-3.1-Tulu-3-8B"
model="unsloth/Llama-3.1-Tulu-3-70B-bnb-4bit"

#model="Qwen/Qwen2.5-32B-Instruct-AWQ"
#model="Qwen/Qwen2.5-1.5B-Instruct"
model="unsloth/Qwen2.5-14B-Instruct"

if [[ "${1}" != "--"* ]]; then
    model_path=${1:-$model}
    shift
fi

if [[ "$model_path" == "Qwen/"* ]]; then
  parser="hermes"
else
  parser="llama3_json"
fi

vllm serve $model_path --enable-auto-tool-choice --tool-call-parser $parser "$@"