# Installation

## Environment (ARM sillicon)

Needed to install tensorflow

```bash
brew install cmake pkg-config
```

## Environment

Recommendation: Use python virutal env or conda

```bash
pip install -r requirements.txt
hugging-face-cli login # (get tokens from: https://huggingface.co/settings/tokens)
```

export paths / add into your shell config (e.g. .bashrc)

```bash
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export BNB_CUDA_VERSION=124
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

build model

```bash
python build.py
```
