#!/bin/bash
pip install -r requirements.txt
python3 << 'PY'
import os
from gpt4all import GPT4All

# Pre-download the model so it's cached in the build
model_name = "orca-mini-3b-gguf2-q4_0.gguf"
print(f"Pre-downloading {model_name}...")
GPT4All(model_name)
print("Model ready!")
PY
