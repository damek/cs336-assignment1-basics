# Twitter thread

I completed this assignment. You can read my thoughts while completing the assignment here: 

https://x.com/damekdavis/status/1937275870663598216

You can also check the changelog!

## Runing a sweep with my code 

```python
# clone the repo (assuming you've done this)
# Set up uv 
cd cs336-assignment1-basics
uv venv 
source .venv/bin/activate
uv pip install -e .
# cd into working directory
cd cs336_basics
# Download data 
uv run data/TinyStories.py
uv run data/OpenWebText32k.py
# Run sweeps 
## Tip: Do not compile on an MPS device.
uv run sweep.py --cfg_cls configs:TSCfg --bs 32 --run_until_step 40000 --device cuda --print_every 100 --compile True cosine --min_lr 1e-7 --max_lr .0025 --warmup_end 400
```

# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

