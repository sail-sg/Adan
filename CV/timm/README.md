# Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models

For vision tasks, our implementation is based on the official [`timm`](https://github.com/rwightman/pytorch-image-models). To reproduce our results, please first refer to [`timm`](https://github.com/rwightman/pytorch-image-models) and install it. Then you can follow the following two steps to reproduce our experiments in paper. 



## Environment

Our experiments for this task are based on the following pkg version.

```python
torch.__version__  = '1.10.0+cu113'
torchvision.__version__ = '0.11.1+cu113'
timm.__version__ = '0.6.1'
torchaudio.__version__ = '0.10.0+cu113'
```

Note that our timm is a developer version. If you want to strictly follow our environment, please refer to our released docker image [xyxie/adan-image:timm](https://hub.docker.com/repository/docker/xyxie/adan-image).



## Usage of Adan in timm

### Two steps to use Adan

**Step 1.** add Adan-dependent hyper-parameters by adding the following hyper-parameters to the `train.py`:

```python
parser.add_argument('--max-grad-norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
parser.add_argument('--weight-decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
parser.add_argument('--no-prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')
parser.add_argument('--bias-decay', action='store_true', default=False, help='Perform the weight decay on bias term (default=False)')

```

* `bias-decay`: It decides whether or not to perform the weight decay on 1) bias term, 2) bn, and 3) other 1d params, which are all filtered out by the default setting in timm.

* `no-prox`: It determines the update rule of parameters with weight decay. By default, Adan updates the parameters in the way presented in Algorithm 1 in the paper:

    $$\boldsymbol{\theta}_{k+1} = ( 1+\lambda \eta)^{-1}\left[\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}_k)\right],$$

  But one also can update the parameter like Adamw:

  $$\boldsymbol{\theta}_{k+1} = ( 1-\lambda \eta)\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}_k).$$
  **In all experiments, we set `no-prox=False` in our paper.** 

  

**Step 2.** creat the Adan optimizer as follows. In this step, we directly replace the vanilla optimizer creator by using the following three substeps. 

i) add Adan into `optim_factory.py`:
  ```python
  elif opt_lower == 'adan': 
    optimizer = Adan(parameters, **opt_args)
  ```

ii) import the optimizer creator into your training file `train.py` from `optim_factory` :

  ```python
  from optim_factory import create_optimizer
  ```

iii) replace the vanilla creator (`optimizer = create_optimizer(args, model)`) in the training file `train.py`  with Adan:

  ```python
  opt_lower = args.opt.lower()
  if opt_lower == 'adan':
    args.opt_args = {'max_grad_norm': args.max_grad_norm, 'no_prox': args.no_prox}
  optimizer = create_optimizer(args, model, filter_bias_and_bn = not args.bias_decay)
  ```



## ImageNet-1K Training Recipe

We provide the specific commonds and hyper-parameters for ViTs, ResNets and ConvNexts in this [recipe](./supervised.md).

