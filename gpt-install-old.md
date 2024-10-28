# Installation

## Environment (ARM sillicon)

```bash
CONDA_SUBDIR=osx-64 conda create -n gpt86 python=3.6.3
conda activate gpt86
conda config --env --set subdir osx-64
```

## Environment (x86)

```bash
conda create -n gpt86 python=3.6.3
conda activate gpt86
```

## Install dependencies

`pip install -r requirements.txt`

## Install prebuilt tensorflow 1.5.1

`pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.5.0-py3-none-any.whl`

## Install gpt model

[!NOTE]
Model options: 117M, 124M, 245M, 255M, 774M, 1558M

`python download_model.py <model options>`

## Test run

`python src/interactive_conditional_samples.py --model_name <model options> --top_k 40`

## If got sorting error

`sudo sed -ie s/sort\(/contrib\.framework\.sort\(/g src/sample.py`

Test run again afterwards

## Reference

[ARM installation](https://www.youtube.com/watch?v=iv07Vtfd_4o)
