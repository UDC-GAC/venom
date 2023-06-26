#!/bin/bash

export YAML_NAME=30epochs_8block875_squad
export RECIPE=integrations/huggingface-transformers/recipes/${YAML_NAME}.yaml

# for 12-layer model: export MODEL=bert-base-uncased
# for 6-layer model: export MODEL=neuralmagic/oBERT-6-upstream-pretrained-dense
# for 3-layer model: export MODEL=neuralmagic/oBERT-3-upstream-pretrained-dense
export MODEL=bert-base-uncased

#uncomment to run on a single-gpu
#CUDA_VISIBLE_DEVICES=0 python3.10 src/sparseml/transformers/question_answering.py \
python3.10 -m torch.distributed.launch --nproc_per_node=3 src/sparseml/transformers/question_answering.py \
  --distill_teacher neuralmagic/oBERT-teacher-squadv1 \
  --model_name_or_path ${MODEL} \
  --dataset_name squad \
  --do_train \
  --fp16 \
  --do_eval \
  --optim adamw_torch \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 8e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --preprocessing_num_workers 8 \
  --seed 42 \
  --num_train_epochs 30 \
  --recipe ${RECIPE} \
  --output_dir integrations/huggingface-transformers/output_dir/block8_875 \
  --overwrite_output_dir \
  --skip_memory_metrics true \
#--report_to wandb
