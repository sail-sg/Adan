# Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models



## Installation of Fairseq

Our experiment is based on the repo [Fairseq](https://github.com/facebookresearch/fairseq). For the requirements and installation of [Fairseq](https://github.com/facebookresearch/fairseq) and Apex, please refer to that repo.



## Environment

Our experiments for this task are based on the following pkg version.

```python
torch.__version__  = '1.10.1+cu111'
torchvision.__version__ = '0.11.2+cu111'
torchaudio.__version__ = '0.10.1+cu111'
fairseq.__version__ = '0.12.2'
```

If you want to strictly follow our environment, please refer to our released docker image [xyxie/adan-image:fairseq](https://hub.docker.com/repository/docker/xyxie/adan-image).



## Usage of Adan in Fairseq

### One step to use Adan

Please first put the file [`adan.py`](./adan.py) to the directory `path/to/fairseq/fairseq/optim`. Then you can choose Adan as the optimizer in the config file. See  following example for pre-training:

```yaml
optimizer:
  _name: adan
  weight_decay: 0.02
  adan_betas: (0.98,0.92,0.99)
  adan_eps: 1e-08
```



## Pretraining

The following steps are modified from [Fairseq-Roberta](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md). For completeness, we list some key steps here.


### 1) Preprocess the data

Data should be preprocessed following the [language modeling format](https://github.com/facebookresearch/fairseq/tree/main/examples/language_model). That is, each document should be separated by an empty line (only useful with `--sample-break-mode complete_doc`, and all lines should be concatenated as a 1D text stream during training.



In the following steps, we use the [Bookcorpus dataset](https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz) and [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download) to demonstrate how to preprocess raw text data with the GPT-2 BPE.

#### i) Download the dataset:

```bash
wget https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz
tar  -zxvf  books1.tar.gz  -C  ./bert-corpus/
```

```python
pip install datasets
from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.en")
```

#### ii) Generate Raw data:

   - For wikipedia dataset,  we need to read each line of the json line file , replace the `\n` in the text field with a space, and write the line (add `\n` at the end), to the file new  `all_data.raw`.

   - For  bookcorpus dataset, read out the contexts of each book, then replace  the `\n` with the space, and then write the context of the book as one line in `all_data.raw`, ended up with `\n`.

   - Split the  `all_data.raw`  in to  `wiki.train.raw` and  `wiki.dev.raw`  with the ratio of 99:1. Set  `wiki.test.raw = wiki.dev.raw` for compatibility of fairseq.

     

#### iii) Encode data with the GPT-2 BPE:

```bash
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs bert-corpus/wiki.${SPLIT}.raw \
        --outputs bert-corpus/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```



#### iv) Binarize the data using the GPT-2 fairseq dictionary:

```bash
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref bert-corpus/wiki.train.bpe \
    --validpref bert-corpus/wiki.valid.bpe \
    --testpref bert-corpus/wiki.test.bpe \
    --destdir data-bin/bert-corpus \
    --workers 60
```



### 2) Train BERT base

Put the provided [config files](./config/pretraining) to the directory `path/to/fairseq/examples/roberta/config/pretraining`

```bash
DATA_DIR=/path/to/fairseq/bert-corpus

fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name ${NAME} task.data=$DATA_DIR \
checkpoint.save_dir=/path/to/save_dir/

```

We can optionally resume the training of the released BERT-base model by adding `checkpoint.restore_file=/path/to/model.pt`. Note, in our experiments, we use Adan to train BERT-base from scratch. You can use the following config files to train  BERT-base with Adam or Adan:

  |   NAME    | Optimizer |                         Config                         |                         Download                         |
  | :-------: | :-------: | :----------------------------------------------------: | :------------------------------------------------------: |
  | bert-base |   Adam    | [config](./exp_results/pretrain/full_config-adam.yaml) | [log](./exp_results/pretrain/hydra_train-adam.log)/model |
  | bert-adan |   Adan    | [config](./exp_results/pretrain/full_config-adan.yaml) | [log](./exp_results/pretrain/hydra_train-adan.log)/model |

The above command assumes the training is on 8x40GB A100 GPUs. Each GPU uses a batch size of 32 sequences (`dataset.batch_size`). If you have fewer GPUs or GPUs with less memory, you may need to reduce `dataset.batch_size` and increase `dataset.update_freq` to compensate. Alternatively if you have more GPUs you can decrease `dataset.update_freq` accordingly to improve the training speed.


## Finetuning BERT-base on GLUE tasks

### 1) Download the data from [GLUE website](https://gluebenchmark.com/tasks) using following commands:
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```
There some problems to download `MRPC` and  `MNLI` , hence we pass the `MRPC` task and download the data of `MNLI` from the unofficial sources.



### 2) Preprocess GLUE task data:

```bash
./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
```
- `glue_task_name` is one of the following: `{ALL, QQP, MNLI, QNLI, RTE, STS-B, SST-2, CoLA}`. Use `ALL` for preprocessing all the glue tasks.



### 3) Fine-tuning on GLUE task:

Example fine-tuning cmd for `RTE` task
```bash
TASK=RTE;

python  path/to/fairseq/examples/roberta/config/finetuning/acc_test.py --avg_num 1 \
--data_path /path/to/fairseq/GLUE/glue_data/$TASK \
--bin_path /path/to/fairseq/GLUE/$TASK-bin \
--pre_path /path/to/fairseq/bert-adan/checkpoint_best.pt \
--finetune_path /path/to/fairseq/bert-fintune/adan/$TASK/ \
--task rte-adan
```

- `avg_num` number of repetitions.

- `data_path` path to the data of GLUE task, e.g., CoLA, MNLI, etc.

- `bin_path` similar to `data_path`, but is path to the binarized data after processing.

- `pre_path` path to the pre-trained model.

- `finetune_path` path to save/load fine-tuned model.

- `task` config name, please refer to the directory of [fine-tuning](./config/finetuning) for the additional config files for each of the GLUE tasks.

- This cmd-args and hyperparams are tested on one Nvidia `A100` GPU with `40gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--batch-size`.

  

### 4) Inference on GLUE task
After training the model by using previous step, we can perform inference with checkpoints in `finetune_path` directory using following code snippet:

```bash
TASK=RTE;

python  path/to/fairseq/examples/roberta/config/finetuning/acc_test.py --inference \
--data_path /path/to/fairseq/GLUE/glue_data/$TASK \
--bin_path /path/to/fairseq/GLUE/$TASK-bin \
--pre_path /path/to/fairseq/bert-adan/checkpoint_best.pt \
--finetune_path /path/to/fairseq/bert-fintune/adan/$TASK/ \
--task rte-adan

```

 This should give:

| GLUE-Task | Metric                       |  Result   |                    Config                     |
| --------- | :--------------------------- | :-------: | :-------------------------------------------: |
| CoLA      | Matthew's corr.              |   64.6    | [config](./config/finetuning/cola-adan.yaml)  |
| SST-2     | Accuracy                     |   93.2    | [config](./config/finetuning/sst_2-adan.yaml) |
| STS-B     | Person corr.                 |   89.3    | [config](./config/finetuning/sts_b-adan.yaml) |
| QQP       | Accuracy                     |   91.2    |  [config](./config/finetuning/qqp-adan.yaml)  |
| MNLI      | Matched acc./Mismatched acc. | 85.7/85.6 | [config](./config/finetuning/mnli-adan.yaml)  |
| QNLI      | Accuracy                     |   91.3    | [config](./config/finetuning/qnli-adan.yaml)  |
| RTE       | Accuracy                     |   73.3    |  [config](./config/finetuning/rte-adan.yaml)  |

