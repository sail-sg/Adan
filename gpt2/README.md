# Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models

This experiment is based on the warped repo of [Megatron-LM](https://github.com/bigcode-project/Megatron-LM), provided by [BigCode](https://www.bigcode-project.org/). And the task of this experiment is code generalization.

## Usage of Adan in Megatron-LM

### Two steps to use Adan

**Step 1.** put `adan.py` in the path `Megatron-LM/megatron/optimizer/adan.py` and import it in the `Megatron-LM/megatron/optimizer/__init__.py`.

```python
from .adan import Adan

elif args.optimizer == 'adan':
  optimizer = Adan(param_groups,lr=args.lr, weight_decay=args.weight_decay,
                   betas=(args.adan_beta1, args.adan_beta2, args.adan_beta3),
                   eps=args.adan_eps)
```

**Step 2.** add the following parameters to the file `Megatron-LM/megatron/arguments.py`.

```python
# beta3 is for the optimizer Adan, but not used in Adam.
group.add_argument('--adan-beta1', type=float, default=0.98,
                   help='First coefficient for computing running averages '
                   'of gradient and its square')
group.add_argument('--adan-beta2', type=float, default=0.92,
                   help='Second coefficient for computing running averages '
                   'of gradient and its square')
group.add_argument('--adan-beta3', type=float, default=0.99,
                   help='Second coefficient for computing running averages '
                   'of gradient and its square')
group.add_argument('--adan-eps', type=float, default=1e-08,
                   help='Term added to the denominator to improve'
                   'numerical stability')
group.add_argument('--optimizer', type=str, default='adam',
                   choices=['adam', 'sgd', 'adan'],
```

- `adan-beta1,2,3`: optimizer betas for Adan.

- `adan-eps`: stabilizing parameter.

- `optimizer`: choices of optimizers.

## Data Preparation

**Step 1.** download the dataset used for pre-training. The dataset is collected and released by [BigCode](https://www.bigcode-project.org/) project:

```python
python ./download_dataset.py
```

**Step 2.** binarize the downloaded dataset:

```python
python tools/preprocess_data.py \
      --input stack_python.json \
      --output-prefix codegpt \
      --vocab checkpoints/gpt2-adan/tokenizer/vocab.json \
      --json-key content \
      --dataset-impl mmap \
      --workers 16 \
      --chunk-size 25 \
      --tokenizer-type GPT2BPETokenizer \
      --merge-file checkpoints/gpt2-adan/tokenizer/merges.txt \
      --append-eod; \
```

## Pre-training

- #### Installation and Export

  ```bash
  pip install wandb; \
  pip install regex; \
  pip install pybind11; \
  pip install nltk; \
  export MASTER_NODE=localhost; \
  export NUM_NODES=8; \
  export NODE_RANK=0; \
  export WANDB_API_KEY=$YOUR_API; \
  export WANDB_NAME=$PROJECT_NAME; \
  export WANDB_NOTES=$NOTES; \
  ```

- #### Training

  `bash ./pretrain.sh`

## Results and Logs on GPT2-345m

We provide the config and log for GPT2-345m pre-trained on the dataset that comes from [BigCode](https://www.bigcode-project.org/) and evaluated on the [HumanEval](https://github.com/openai/human-eval) dataset by zero-shot learning. [HumanEval](https://github.com/openai/human-eval) is used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions. We set ` Temperature = 0.8` during evaluation.

|                  | Steps | pass@1 | pass@10 | pass@100 |                                  Download                                  |
| ---------------- | :---: | :----: | :-----: | :------: | :------------------------------------------------------------------------: |
| GPT2-345m (Adam) | 300k  | 0.0840 |  0.209  |  0.360   | [log&config](https://github.com/sail-sg/Adan/files/10362486/gpt2-adam.log) |
| GPT2-345m (Adan) | 150k  | 0.0843 |  0.221  |  0.377   | [log&config](https://github.com/sail-sg/Adan/files/10362485/gpt2-adan.log) |
