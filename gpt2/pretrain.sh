set -u # stop on unset variables

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_NODE}
MASTER_PORT=6000
NNODES=${NUM_NODES}
# NODE_RANK=0  # env
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_NAME=gpt2-adan
CHECKPOINT_PATH=checkpoints/$CHECKPOINT_NAME  # Directory to store the checkpoints
PREPROCESSED_DATA=preprocessed  # Directory containing the preprocessed dataset. To preprocess a dataset, see https://github.com/bigcode-project/Megatron-LM#data-preprocessing
VOCAB_FILE=${CHECKPOINT_PATH}/tokenizer/vocab.json
MERGE_FILE=${CHECKPOINT_PATH}/tokenizer/merges.txt
DATA_PATH=${PREPROCESSED_DATA}/codegpt_content_document

GPT_ARGS=$(cat ${CHECKPOINT_PATH}/gpt_args)

TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"

python -m torch.distributed.launch --nproc_per_node=8 \
       pretrain_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --recompute-activations \
       $GPT_ARGS \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --wandb-entity-name xyxie \
       --wandb-project-name $WANDB_NAME \
       $TENSORBOARD_ARGS
       # Uncomment the next two lines to finetune from a pretrained model.
       # --finetune \
       # --finetune-from /directory/containing/pretrained/model
