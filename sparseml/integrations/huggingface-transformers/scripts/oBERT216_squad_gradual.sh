#!/bin/bash

export YAML_NAME=oBERT_216_gradual
export RECIPE=integrations/huggingface-transformers/recipes/${YAML_NAME}.yaml

#uncomment to run on a single-gpu
#CUDA_VISIBLE_DEVICES=0 python3.10 src/sparseml/transformers/question_answering.py \
python3.10 -m torch.distributed.launch --nproc_per_node=3 src/sparseml/transformers/question_answering.py \
--distill_teacher neuralmagic/oBERT-teacher-squadv1 \
--model_name_or_path bert-base-uncased \
--dataset_name squad \
--do_train \
--fp16 \
--do_eval \
--optim adamw_torch \
--evaluation_strategy epoch \
--save_strategy epoch \
--save_total_limit 2 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 8e-5 \
--max_seq_length 384 \
--doc_stride 128 \
--preprocessing_num_workers 8 \
--seed 42 \
--num_train_epochs 50 \
--recipe ${RECIPE} \
--output_dir integrations/huggingface-transformers/output_dir/test_obert716_gradual \
--overwrite_output_dir \
--skip_memory_metrics true \
--report_to none \