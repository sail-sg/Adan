# Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models

We first provide the instruction to modify the official training files from [Transformer-XL](https://github.com/kimiyoung/transformer-xl) to support Adan. **For data preparation, please follow that repo.**

## Environment

As recommended by the official [Transformer-XL](https://github.com/kimiyoung/transformer-xl), our experiments for this task are based on the following pkg version.

```python
torch.__version__  = '1.1.0'
```

## Usage of Adan for Transformer-XL

### Two steps to use Adan

**Step 1.** add the following parameters to the file `train.py`.

```python
parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd', 'adagrad', 'adan'], help='optimizer to use.')
parser.add_argument('--wd', type=float, default=0.02, help='weight decay (default: 0.02)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
```

- `optim`: the choice of optimizers. We add Adan in the choices.

- `wd`: decoupled weight decay.

- `opt-betas`: optimizer betas for Adan.

**Step 2.** replace the original optimizitor creation with the following:

```python
from adan import Adan

elif args.optim.lower() == 'adan':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = Adan(sparse_params,betas=args.opt_betas, lr=args.lr, weight_decay= args.wd)
        optimizer = Adan(dense_params, lr=args.lr,betas=args.opt_betas, weight_decay= args.wd)
    else:
        optimizer = Adan(model.parameters(), lr=args.lr, betas=args.opt_betas, weight_decay= args.wd)

```

## Data Preparation

see `bash getdata.sh` in repo  [Transformer-XL](https://github.com/kimiyoung/transformer-xl).

## Training and Evaluation

- #### Training

  `bash run_wt103_adan.sh train --work_dir PATH_TO_WORK_DIR`

- #### Evaluation

  `bash run_wt103_adan.sh eval --work_dir PATH_TO_WORK_DIR`

- #### Tips for Experiments

  - For Adan, we set `args.wd = 0.02` for all steps, which is consistent with the other experiments.
  - For the experiment using `steps = 50k`, we choose a slightly larger `LR`.

## Results and Logs

With a different setting for `lr` and `max_step` in `run_wt103_adan.sh`, we have the following results:

|                     |   LR   | Steps | Test PPL |                 Download                 |
| ------------------- | :----: | :---: | :------: | :--------------------------------------: |
| Baseline (Adam)     | 2.5e-4 | 200k  |   24.2   | [log&config](./exp_results/log-adam.txt) |
| Transformer-XL-base | 1.5e-3 |  50k  |   26.2   | [log&config](./exp_results/log-50k.txt)  |
| Transformer-XL-base |  1e-3  | 100k  |   24.2   | [log&config](./exp_results/log-100k.txt) |
| Transformer-XL-base |  1e-3  | 200k  |   23.5   | [log&config](./exp_results/log-200k.txt) |
