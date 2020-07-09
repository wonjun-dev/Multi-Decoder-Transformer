#!/bin/bash

dataset=chembl_lm
latent=(0)
dropout=(0.25)
model_name=pretrain_masked_lm
device=$1
seed=$2


for l in ${latent[@]}; do
    for p in ${dropout[@]}; do
        # Train
        CUDA_VISIBLE_DEVICES="${device}" python  train.py -data data/${dataset}/${dataset} \
                -save_model experiments/${dataset}_${model_name}_${l}_${p}_${seed} \
                -gpu_ranks 0 -save_checkpoint_steps 1000  -keep_checkpoint 16 \
                -train_steps 200000 -valid_steps 1000 -report_every 1000 -param_init 0  -param_init_glorot \
                -batch_size 32 -batch_type sents -normalization sents -max_grad_norm 0  -accum_count 4 \
                -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                -learning_rate 2 -label_smoothing 0.0 \
                -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                -dropout "${p}" -position_encoding -share_embeddings -pretrain_masked_lm \
                -heads 8 -transformer_ff 2048 -tensorboard -tensorboard_log_dir runs/${dataset}_${model_name} -world_size=1 -seed "${seed}" -early_stopping 15 -n_latent "${l}"

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
