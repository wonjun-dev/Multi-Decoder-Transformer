#!/bin/bash

dataset=hiv
#feat_names=(NR-AhR NR-AR-LBD NR-AR NR-Aromatase NR-ER-LBD NR-ER NR-PPAR-gamma
#SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53)
#feat_name=${feat_names[3]}
latent=(0) # 2 5)
dropout=(0.25)
device=$1
seed=$2
model_dim=256 #$3 # 256
ff_dim=512 #$4 # 2048
n_head=8 #$5 # 8
model_name=mt-d${model_dim}-ff${ff_dim}-h${n_head}-pretrained #neighbor_attr_pretrained_mt #_freeze_enc_2000 #_noshareembed
#model_name=mt #neighbor_attr_contrastive_pretrained_mt #_freeze_enc_2000 #_noshareembed


#python3 preprocess_xy.py # split 다시하기
for l in ${latent[@]}; do
    for p in ${dropout[@]}; do
        # Train
        CUDA_VISIBLE_DEVICES="${device}" python  train.py -data data/${dataset}/${dataset}-${seed} \
                -save_model experiments/${dataset}_${model_name}_${l}_${p}_${seed} \
                -gpu_ranks 0 -save_checkpoint_steps 1000  -keep_checkpoint 16 \
                -train_steps 200000 -valid_steps 1000 -report_every 1000 -param_init 0  -param_init_glorot \
                -batch_size 32 -batch_type sents -normalization sents -max_grad_norm 0  -accum_count 1 \
                -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 4000  \
                -learning_rate 0.1 -label_smoothing 0.0 \
                -layers 6 -rnn_size ${model_dim} -word_vec_size ${model_dim} -encoder_type transformer -decoder_type transformer \
                -dropout "${p}" -position_encoding \
                -share_embeddings -binary_clf \
                -heads ${n_head} -transformer_ff ${ff_dim} \
                -early_stopping_criteria auc \
                -update_enc_from 0 \
                -n_gen_layer 2 \
                -pool_factor 1 \
                -world_size=1 -seed "${seed}" -early_stopping 100 -n_latent "${l}" \
                -train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive_toksim-6-512-256-0.1_0_0.1_2021/models/model_step_292000.pt
                #-start_decay_steps 40000 -decay_steps 40000 -learning_rate_decay 0.1
# experiments/chembl_lm_pretrain_neighbor_attr_contrastive_toksim-4-2048-256-0.1_0_0.1_2022/models/model_step_93000.pt
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive_toksim-6-512-256-0.1_0_0.1_2021/models/model_step_168000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive-0.1_0_0.1_2021/models/model_step_200000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive-0.1_0_0.1_2022/models/model_step_81000.pt
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive-0.1_0_0.1_2021/models/model_step_9000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr__0_0.25_2021/models/model_step_51000.pt
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr__0_0.25_2020/models/model_step_6000.pt
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_0_0.25_2020/models/model_step_6000.pt
                #-tensorboard -tensorboard_log_dir runs/${dataset}_${model_name} \
                #-train_from experiments/chembl_lm_pretrain_neighbor_0_0.25_2020/models/model_step_125000.pt \
                #-update_enc_from 2000 \
    done
done
