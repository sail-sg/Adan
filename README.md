# Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models

This is an official PyTorch implementation of **Adan**. See paper [here](https://arxiv.org/abs/2208.06677). If you find our adan helpful or heuristic to your projects, please cite this paper and also star this repository. Thanks!




```tex
@article{xie2022adan,
  title={Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models},
  author={Xie, Xingyu and Zhou, Pan and Li, Huan and Lin, Zhouchen and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2208.06677},
  year={2022}
}
```



---
+ :fire: :fire: Faster implementation are released. 

+ Adan is supported in the lasted version of [`timm`](https://github.com/rwightman/pytorch-image-models).
+ TF's implementation (third party) refers to [DenisVorotyntsev/Adan](https://github.com/DenisVorotyntsev/Adan).
+ JAX's version (third party) is implemented and also supported in [Deepmind/optax](https://github.com/deepmind/optax).

---



## Usage

For your convenience to use Adan, we briefly provide some intuitive instructions below, then provide some general experimental tips, and finally give more details (e.g. specific commonds and hyper-parameters) for each experiment in the paper. 

#### 1) Two steps to use Adan

**Step 1.** add Adan-dependent hyper-parameters by adding the following hyper-parameters to the config:

```python
parser.add_argument('--max-grad-norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
parser.add_argument('--weight-decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
parser.add_argument('--no-prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')
```
`opt-betas` In order to keep consistent with our usage habits, the $\beta$'s in the paper is actually the $(1-\beta)$'s in the code.

`no-prox`: It determines the update rule of parameters with weight decay. By default, Adan updates the parameters in the way presented in Algorithm 1 in the paper:

  $$\boldsymbol{\theta}_{k+1} = ( 1+\lambda \eta)^{-1}\left[\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}_k)\right],$$_

But one also can update the parameter like Adamw:

$$\boldsymbol{\theta}_{k+1} = ( 1-\lambda \eta)\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}_k).$$
In all experiments, we set `no-prox=False` in our paper. 

**Step 2.** creat the Adan optimizer as follows. In this step, we can directly replace the vanilla optimizer by using the following command:

```python
from adan import Adan
optimizer = Adan(param, lr=args.lr, weight_decay=args.weight_decay, betas=args.opt_betas, eps = args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)
```

#### 2) Tips for Experiments

- To make Adan simple, in all experiments except Table 12 in the paper, we do not use the restart strategy in Adan. But Table 12 shows that restart strategy can further slightly improve  the performance of Adan.
- Adan often allow one to use a large peak learning rate which often fails other optimizers, e.g. Adam and AdamW. For example, in all experiments except for the experiments on MAE pre-training and LSTM, the learning rate used by Adan is **5-10 times** than that in Adam/AdamW.
- For vision tasks, but not for NLP and RL tasks, it seems that Adan prefers a large batch size for large-scale experiments, e.g. 2,048 total batch size in our paper. 
- Adan is relatively robust to `beta1`, `beta2` and `beta3`, especially for `beta2`. If you hope better performance, you can first tune `beta3` and then `beta1`.  
- Interestingly, we found that `weight_decay = 0.02` is suitable for all experiments in our paper.
- In your training, if you already fully use the GPU memory when using other optimizers, then you may need to scale the batch size slightly smaller (accordingly linearly scaling LR, etc.) when using Adan, since Adan has slightly higher GPU memory cost than others.

#### 3) More extra detailed steps to reproduce experimental results in paper 

Please refer to the following links for detailed steps. In these detailed steps, we even include the **docker images** for reproducibility. 

- [Instruction](./CV/timm/) for **<u>ViTs</u>**, **<u>ResNets</u>**, and **<u>ConvNext</u>**.
- [Instruction](./CV/MAE/) for **<u>MAE</u>**.
- [Instruction](./NLP/BERT/) for **<u>BERT</u>**.
- [Instruction](./NLP/Transformer-XL/) for **<u>Transformer-XL</u>**.



## Model Zoo

### Results on vision tasks

For your convenience to use Adan, we provide the configs and log files for the experiments on ImageNet-1k.

| Model         |  Epoch  | Training Setting | Acc. (%) |                            Config                            |                            Batch Size                            |                           Download                           |
| ------------- | :-----: | :-----: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ViT-S         |   150   |    I    |   80.1   | [config](./CV/timm/exp_results/ViT/small/args_vit-s_150-I.yaml) | 2048 | [log](./CV/timm/exp_results/ViT/small/summary_vit-s_150-I.csv)/model |
| ViT-S         |   150   |   II    |   79.6   | [config](./CV/timm/exp_results/ViT/small/args_vit-s_150.yaml) | 2048 |  [log](./CV/timm/exp_results/ViT/small/summary_vit-s_150.csv)/model  |
| ViT-S         |   300   |    I    |   81.1   | [config](./CV/timm/exp_results/ViT/small/args_vit-s_300-I.yaml) | 2048 | [log](./CV/timm/exp_results/ViT/small/summary_vit-s_300-I.csv)/model |
| ViT-S         |   300   |   II    |   80.7   | [config](./CV/timm/exp_results/ViT/small/args_vit-s_300.yaml) | 2048 | [log](./CV/timm/exp_results/ViT/small/summary_vit-s_300.csv)/model |
| ViT-B         |   150   |   II    |   81.7   | [config](./CV/timm/exp_results/ViT/base/args_vit-B_150.yaml) | 2048 | [log](./CV/timm/exp_results/ViT/base/summary_vit-B_150.csv)/model |
| ViT-B         |   300   |   II    |   82.6   | [config](./CV/timm/exp_results/ViT/base/args_vit-B_300_T.yaml) | 2048 | [log](./CV/timm/exp_results/ViT/base/summary_vit-B_300_T.csv)/model |
| ResNet-50     |   100   |    I    |   78.1   | [config](./CV/timm/exp_results/ResNet/Res50/args_res50_100.yaml) | 2048 | [log](./CV/timm/exp_results/ResNet/Res50/summary_res50_100.csv)/model |
| ResNet-50     |   200   |    I    |   79.7   |   [config](./CV/timm/exp_results/ResNet/Res50/args_res50_200.yaml)   |   2048   | [log](./CV/timm/exp_results/ResNet/Res50/summary_res50_200.csv)/model |
| ResNet-50     |   300   |    I    |   80.2   | [config](./CV/timm/exp_results/ResNet/Res50/args_res50_300.yaml) | 2048 | [log](./CV/timm/exp_results/ResNet/Res50/summary_res50_300.csv)/model |
| ResNet-101 | 100 | I | 80.0 | [config](./CV/timm/exp_results/ResNet/Res101/args_res101_100.yaml) | 2048 | [log](./CV/timm/exp_results/ResNet/Res101/summary_res101_100.csv)/model |
| ResNet-101 | 200 | I | 81.6 | [config](./CV/timm/exp_results/ResNet/Res101/args_res101_200.yaml) | 2048 | [log](./CV/timm/exp_results/ResNet/Res101/summary_res101_200.csv)/model |
| ResNet-101 | 300 | I | 81.9 | [config](./CV/timm/exp_results/ResNet/Res101/args_res101_300.yaml) | 2048 | [log](./CV/timm/exp_results/ResNet/Res101/summary_res101_300.csv)/model |
| ConvNext-tiny |   150   |   II    |   81.7   | [config](./CV/timm/exp_results/ConvNext/small/args_cvnext_150.yaml) | 2048 | [log](./CV/timm/exp_results/ConvNext/small/summary_cvnext_150.csv)//model |
| ConvNext-tiny |   300   |   II    |   82.4   | [config](./CV/timm/exp_results/ConvNext/small/args_cvnext_300.yaml) | 2048 | [log](./CV/timm/exp_results/ConvNext/small/summary_cvnext_300.csv)/model |
| MAE-small     | 800+100 |   ---   |   83.8   |                 [config](./CV/MAE/README.md)                 |                 4096/2048                 | [log-pretrain](./CV/MAE/exp_results/MAE/base/log_base_pretrain.txt)/[log-finetune](./CV/MAE/exp_results/MAE/base/log_base_ft.txt)/model |
| MAE-Large     | 800+50  |   ---   |   85.9   |                 [config](./CV/MAE/README.md)                 |                 4096/2048                 | [log-pretrain](./CV/MAE/exp_results/MAE/large/log_large_pretrain.txt)/[log-finetune](./CV/MAE/exp_results/MAE/large/log_large_ft.txt)/model |



### Results on NLP tasks

#### BERT-base

We give the configs and log files of the BERT-base model pre-trained on the Bookcorpus and Wikipedia datasets and fine-tuned on GLUE tasks. Note, we provide the config and log file, and detailed [instruction](./NLP/BERT/README.md) for BERT-base in the folder `./NLP/BERT`.




| Pretraining | Config  | Batch Size |  Log   | Model  |
| --------- | :--------: | :--------: | :--------: | :--------: |
| Adan      |  [config](./NLP/BERT/config/pretraining/bert-adan.yaml)  |  256  |   [log](./NLP/BERT/exp_results/pretrain/hydra_train-adan.log)   | model |


| Fine-tuning on GLUE-Task | Metric                       |  Result   |                         Config                          |
| -------------- | :--------------------------- | :-------: | :-----------------------------------------------------: |
| CoLA      | Matthew's corr.              |   64.6    | [config](./NLP/BERT/config/finetuning/cola-adan.yaml)  |
| SST-2     | Accuracy                     |   93.2    | [config](./NLP/BERT/config/finetuning/sst_2-adan.yaml) |
| STS-B     | Person corr.                 |   89.3    | [config](./NLP/BERT/config/finetuning/sts_b-adan.yaml) |
| QQP       | Accuracy                     |   91.2    |  [config](./NLP/BERT/config/finetuning/qqp-adan.yaml)  |
| MNLI      | Matched acc./Mismatched acc. | 85.7/85.6 | [config](./NLP/BERT/config/finetuning/mnli-adan.yaml)  |
| QNLI      | Accuracy                     |   91.3    |  [config](./NLP/BERT/config/finetuning/qnli-adan.yaml)  |
| RTE       | Accuracy                     |   73.3    |  [config](./NLP/BERT/config/finetuning/rte-adan.yaml)   |

For fine-tuning on GLUE-Task, see the total batch size in their corresponding configure files.



#### Transformer-XL-base 

We provide the config and log for Transformer-XL-base trained on the WikiText-103 dataset. The total batch size for this experiment is `60*4`.

|                     | Steps | Test PPL |                          Download                           |
| ------------------- | :---: | :------: | :---------------------------------------------------------: |
| Baseline (Adam)     | 200k  |   24.2   | [log&config](./NLP/Transformer-XL/exp_results/log-adam.txt) |
| Transformer-XL-base |  50k  |   26.2   | [log&config](./NLP/Transformer-XL/exp_results/log-50k.txt)  |
| Transformer-XL-base | 100k  |   24.2   | [log&config](./NLP/Transformer-XL/exp_results/log-100k.txt) |
| Transformer-XL-base | 200k  |   23.5   | [log&config](./NLP/Transformer-XL/exp_results/log-200k.txt) |

  

​	



