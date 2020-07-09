#!/bin/bash

dataset=tox21
feat_names=(NR-AhR NR-AR-LBD NR-AR NR-Aromatase NR-ER-LBD NR-ER NR-PPAR-gamma
SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53)
#feat_name=${feat_names[3]}
latent=(0) # 2 5)
dropout=(0.1 0.25 0.5)
model_name=mt #neighbor_attr_pretrained_mt #_freeze_enc_2000 #_noshareembed
device=$1
seed=$2


#python3 preprocess_xy.py # split 다시하기
for feat_name in ${feat_names[@]}; do
 for l in ${latent[@]}; do
    for p in ${dropout[@]}; do
        # Train
        CUDA_VISIBLE_DEVICES="${device}" python  train.py -data data/${dataset}/${dataset}-${feat_name} \
                -save_model experiments/${dataset}_${feat_name}_${model_name}_${l}_${p}_${seed} \
                -gpu_ranks 0 -save_checkpoint_steps 1000  -keep_checkpoint 16 \
                -train_steps 200000 -valid_steps 1000 -report_every 1000 -param_init 0  -param_init_glorot \
                -batch_size 32 -batch_type sents -normalization sents -max_grad_norm 0  -accum_count 1 \
                -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                -learning_rate 2 -label_smoothing 0.0 \
                -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                -dropout "${p}" -position_encoding \
                -share_embeddings -binary_clf \
                -heads 8 -transformer_ff 2048 \
                -early_stopping_criteria auc \
                -update_enc_from 0 \
                -world_size=1 -seed "${seed}" -early_stopping 15 -n_latent "${l}" \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_0_0.25_2021/models/model_step_200000.pt
                #-tensorboard -tensorboard_log_dir runs/${dataset}_${model_name} \
                #-train_from experiments/chembl_lm_pretrain_neighbor_0_0.25_2020/models/model_step_125000.pt \
                #-update_enc_from 2000 \
    done
  done
done


model_name=neighbor_attr_pretrained_mt #_freeze_enc_2000 #_noshareembed

for feat_name in ${feat_names[@]}; do
 for l in ${latent[@]}; do
    for p in ${dropout[@]}; do
        # Train
        CUDA_VISIBLE_DEVICES="${device}" python  train.py -data data/${dataset}/${dataset}-${feat_name} \
                -save_model experiments/${dataset}_${feat_name}_${model_name}_${l}_${p}_${seed} \
                -gpu_ranks 0 -save_checkpoint_steps 1000  -keep_checkpoint 16 \
                -train_steps 200000 -valid_steps 1000 -report_every 1000 -param_init 0  -param_init_glorot \
                -batch_size 32 -batch_type sents -normalization sents -max_grad_norm 0  -accum_count 1 \
                -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                -learning_rate 2 -label_smoothing 0.0 \
                -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                -dropout "${p}" -position_encoding \
                -share_embeddings -binary_clf \
                -heads 8 -transformer_ff 2048 \
                -early_stopping_criteria auc \
                -update_enc_from 0 \
                -world_size=1 -seed "${seed}" -early_stopping 15 -n_latent "${l}" \
                -train_from experiments/chembl_lm_pretrain_neighbor_attr_0_0.25_2021/models/model_step_200000.pt
                #-tensorboard -tensorboard_log_dir runs/${dataset}_${model_name} \
                #-train_from experiments/chembl_lm_pretrain_neighbor_0_0.25_2020/models/model_step_125000.pt \
                #-update_enc_from 2000 \
    done
  done
done

