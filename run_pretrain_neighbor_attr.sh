#!/bin/bash

dataset=chembl_lm #USPTO-50k_no_rxn_aug_2
latent=(0)
dropout=(0.1)
n_layer=6 #4
ff=2048
model_dim=256
model_name=pretrain_neighbor_attr_contrastive_toksim-${n_layer}-${ff}-${model_dim}
device=$1
seed=$2


for l in ${latent[@]}; do
    for p in ${dropout[@]}; do
        # Train
        model_name_=${model_name}-${p}
        CUDA_VISIBLE_DEVICES="${device}" python  train.py -data data/${dataset}/${dataset}-merge2 \
                -save_model experiments/${dataset}_${model_name_}_${l}_${p}_${seed} \
                -gpu_ranks 0 -save_checkpoint_steps 1000  -keep_checkpoint 16 \
                -train_steps 1000000 -valid_steps 1000 -report_every 1000 -param_init 0  -param_init_glorot \
                -batch_size 16 -batch_type sents -normalization sents \
                -max_generator_batches 0 \
                -max_grad_norm 0  -accum_count 1 \
                -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                -learning_rate 2 -label_smoothing 0.0 \
                -layers ${n_layer} -rnn_size ${model_dim} -word_vec_size ${model_dim} -encoder_type transformer -decoder_type transformer \
                -dropout "${p}" -position_encoding -share_embeddings \
                -contrastive -n_smiles_aug 2 -n_gen_layer 2 \
                -pretrain_attr -pretrain_neighbor \
                -pool_factor 1 \
                -early_stopping_criteria accuracy xent \
                -world_size=1 -seed "${seed}" -early_stopping 1000 -n_latent "${l}" \
                -heads 8 -transformer_ff ${ff} #-tensorboard -tensorboard_log_dir runs/${dataset}_${model_name}
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive_toksim-6-512-256-0.1_0_0.1_2021/models/model_step_200000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive_toksim-6-512-256-0.1_0_0.1_2022/models/model_step_16000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr_contrastive-6-512-256-0.1_0_0.1_2022/models/model_step_31000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr__0_0.25_2021/models/model_step_65000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_attr__0_0.25_2020/models/model_step_14000.pt \
                #-train_from experiments/chembl_lm_pretrain_neighbor_0_0.25_2020/models/model_step_45000.pt \


        # Translate
        #summ_path=experiments/${dataset}_${model_name}_${l}_${p}_${seed}/summary.txt
        #line_cnt=0
        #sep_cnt=0
        #while read line; do
        #    if [ ${line_cnt} -eq 0 ]; then
        #        read -ra ADDR <<< "$line"
        #        for i in "${ADDR[@]}"; do
        #            if [ ${sep_cnt} -eq 1 ]; then
        #                best_model_path=$i
        #            fi
        #        (( sep_cnt=${sep_cnt} + 1 ))
        #        done
        #        (( line_cnt=${line_cnt} + 1 ))
        #    fi
        #done < ${summ_path}
        #echo $best_model_path
        #CUDA_VISIBLE_DEVICES=${device} python translate.py -model ${best_model_path} -src data/${dataset}/src-test.txt -output_dir experiments/${dataset}_${model_name}_${l}_${p}_${seed}/preds/ -beam_size=50 -gpu 0 -batch_size=20 -n_best=50 -log_probs -n_translate_latent ${l}
        ## CUDA_VISIBLE_DEVICES=${device} python translate.py -model ${best_model_path} -src data/${dataset}/src-test-dummy.txt -output_dir experiments/${dataset}_${model_name}_${l}_${p}_${seed}/preds_dummy/ -beam_size=10 -gpu 0 -batch_size=100 -n_best=10 -log_probs -n_translate_latent ${l}

        ## Parse and Save "seed.csv"
        #if [ ${l} -eq 0 ]; then
        #    python parse/parse_output.py -input_file experiments/${dataset}_${model_name}_${l}_${p}_${seed}/preds/output -target_file data/${dataset}/tgt-test.txt -beam_size 50 -result_path experiments/results/${dataset}_${model_name}_${l}_${p}/ -seed ${seed}
        #    # python parse/parse_output.py -input_file experiments/${dataset}_${model_name}_${l}_${p}_${seed}/preds_dummy/output -target_file data/${dataset}/tgt-test.txt -beam_size 50
        #elif [ ${l} -eq 2 ]; then
        #    python parse/parse_output.py -input_dir experiments/${dataset}_${model_name}_${l}_${p}_${seed}/preds/ -target_file data/${dataset}/tgt-test.txt -beam_size 50 -n_latent 2 -result_path experiments/results/${dataset}_${model_name}_${l}_${p}/ -seed ${seed}
        #elif [ ${l} -eq 5 ]; then
        #    python parse/parse_output.py -input_dir experiments/${dataset}_${model_name}_${l}_${p}_${seed}/preds/ -target_file data/${dataset}/tgt-test.txt -beam_size 50 -n_latent 5 -result_path experiments/results/${dataset}_${model_name}_${l}_${p}/ -seed ${seed}
        #fi
    done
done

# Check whether all seeds are dene. if done, save "avg.csv"

#for l in ${latent[@]}; do
#    for p in ${dropout[@]}; do
#        python run_average.py -target_dir experiments/results/${dataset}_${model_name}_${l}_${p}/
#    done
#done
