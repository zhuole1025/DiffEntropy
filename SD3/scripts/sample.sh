#!/bin/bash

export HF_HOME="/data4/zl/.cache/huggingface"
export HF_DATASETS_CACHE="/data4/zl/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/data4/zl/.cache/huggingface/models"

python sample_diffusers.py 