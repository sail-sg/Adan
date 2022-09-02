# Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models

We provide the instruction to modify the official training and fine-tuning files used in [MAE](https://github.com/facebookresearch/mae) such that you can use Adan to train MAE. **Please follow MAE instruction to install necessary packages.**



## Environment

Our experiments for this task are based on the following pkg version.

```python
torch.__version__  = '1.7.1+cu110'
torchvision.__version__ = '0.8.2+cu110'
timm.__version__ = '0.4.5'
torchaudio.__version__ = '0.7.2'
```
If you want to strictly follow our environment, please refer to our released docker image [xyxie/adan-image:mae](https://hub.docker.com/repository/docker/xyxie/adan-image).



## Usage of Adan for MAE

### Two steps to use Adan

**Step 1.** add the following parameters to the `main_pretrain.py` and `main_finetune.py`.

```python
parser.add_argument('--use-adan', action='store_true', default=False, help='whether to use Adan')
parser.add_argument('--max-grad-norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
```

* `use-adan`: whether to use Adan. The default optimizer is AdamW.

* `max-grad-norm`: it determines whether to perform gradient clipping. 

* `opt-eps`: optimizer epsilon to avoid the bad case where second-order moment is zero.

* `opt-betas`: optimizer betas for Adan.

  

**Step 2.** creat the Adan optimizer as follows. In this step, you can directly replace the vanilla optimizer creator :

```python
# following timm: set wd as 0 for bias and norm layers
param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
if args.use_adan:
  if args.bias_decay:
    param = model_without_ddp.parameters() 
  else: 
    param = param_groups
    args.weight_decay = 0.0
    optimizer = Adan(param, weight_decay=args.weight_decay,
                     lr=args.lr, betas=args.opt_betas, 
                     eps = args.opt_eps, max_grad_norm=args.max_grad_norm)
  else:
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
```



## MAE Pre-training

```python
python main_pretrain.py \
    --batch_size ${BS} --accum_iter 1 \
    --model ${MODEL_NAME} --norm_pix_loss --mask_ratio 0.75 \
    --epochs 800 \
    --lr ${LR}  --weight_decay 0.02 --warmup_epochs ${WR_EPOCH} \
    --min_lr ${MIN_LR} \
    --opt-betas 0.98 0.92 0.90 --opt-eps 1e-8 --max-grad-norm 10.0 \
    --use-adan  \
    --data_path ${IMAGENET_DIR}
    --output_dir ${OUT_DIR}
```

- The pre-training file `main_pretrain.py` comes from [MAE](https://github.com/facebookresearch/mae).
- We use **16** A100 GPUs for MAE-Base and **32** A100 GPUs for MAE-Large.
- There are some differences between hyper-parameters for MAE-Base and MAE-Large

|           |      MODEL_NAME       |   LR   |  BS  | MIN_LR | WR_EPOCH |
| :-------: | :-------------------: | :----: | :--: | :----: | :------: |
| MAE-Base  | mae_vit_base_patch16  | 2.0e-3 | 256  |  1e-8  |    40    |
| MAE-Large | mae_vit_large_patch16 | 2.2e-3 | 128  |  1e-4  |    80    |



## MAE Fine-tuning

```python
python main_finetune.py \
  --accum_iter 1 \
  --batch_size ${BS} \
  --model ${MODEL_NAME} \
  --finetune  ${PATH to Ptr-trained Model} \
  --epochs ${EPOCH} \
  --lr 1.5e-2 --layer_decay ${LAYER_DECAY} \
  --min-lr ${MIN_LR} \
  --opt-betas 0.98 0.92 0.99 \
  --opt-eps 1e-8 --max-grad-norm 0 \
  --use-adan --warmup-epochs ${WR_EPOCH} \
  --weight_decay ${WD} --drop_path ${DROP_PATH} \
  --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
  --dist_eval --data_path ${IMAGENET_DIR}
```

- The fine-tune file `main_finetune.py` comes from [MAE](https://github.com/facebookresearch/mae).
- We use **16** A100 GPUs for MAE-Base and **32** A100 GPUs for MAE-Large.
- There are some differences between hyper-parameters for MAE-Base and MAE-Large

|           |    MODEL_NAME     | EPOCH | MIN_LR |  BS  | LAYER_DECAY | WR_EPOCH | WD   | DROP_PATH |
| :-------: | :---------------: | :---: | :----: | :--: | :---------: | :------: | ---- | :-------: |
| MAE-Base  | vit_base_patch16  |  100  |  1e-6  | 128  |    0.65     |    40    | 5e-3 |    0.1    |
| MAE-Large | vit_large_patch16 |  50   |  1e-5  |  64  |    0.75     |    10    | 1e-3 |    0.2    |



## Results and Logs

|          |                           MAE-Base                           |                          MAE-Large                           |
| :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Top-1 Acc. (%) |                             83.8                             |                             85.9                             |
| download | [log-pretrain](./exp_results/MAE/base/log_base_pretrain.txt)/[log-finetune](./exp_results/MAE/base/log_base_ft.txt)/model | [log-pretrain](./exp_results/MAE/large/log_large_pretrain.txt)/[log-finetune](./exp_results/MAE/large/log_large_ft.txt)/model |

