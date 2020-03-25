# Master -> MT Baseline
# This branch is for test set ensemble in MT baseline

Data and preprocessing can be found: https://www.dropbox.com/s/vuhv2a7kgbjsi0s/data.zip?dl=0

This project is built on top of OpenNMT: https://github.com/OpenNMT/OpenNMT-py

To preprocess the data:

```bash
dataset=data_name
python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
```

To train the model:

```bash
dataset=data_name
model_name=model_name
CUDA_VISIBLE_DEVICES=0 python  train.py -data data/${dataset}/${dataset} \
                 -save_model experiments/${dataset}_${model_name} \
                 -gpu_ranks 0 -save_checkpoint_steps 1000  -keep_checkpoint 16 \
                 -train_steps 200000 -valid_steps 1000 -report_every 500 -param_init 0  -param_init_glorot \
                 -batch_size 64 -batch_type sents -normalization sents -max_grad_norm 0  -accum_count 4 \
                 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                 -learning_rate 2 -label_smoothing 0.0 \
                 -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                 -dropout 0.1 -position_encoding -share_embeddings \
                 -heads 8 -transformer_ff 2048 -n_latent 2 -tensorboard -tensorboard_log_dir runs/${dataset}_${model_name} -world_size=1 -seed 2020 -early_stopping 15 
```

To translate:
```bash
python translate.py -model experiments/${dataset}_${model_name}/models/model_step_3.pt -src data/${dataset}/src-test.txt -output_dir experiments/${dataset}_${model_name}/preds/ -replace_unk -beam_size=10 -batch_size=100 -n_best=10 -gpu 0 -log_probs -n_translate_latent 0
```

To test the output results:

```bash
dataset=data_name
N >1:
python parse/parse_output.py -input_dir model_output_file -target_file data/${dataset}/tgt-test.txt -beam_size 10 -n_latent 2 


N = 1
python parse/parse_output.py -input_file model_output_file -target_file data/${dataset}/tgt-test.txt -beam_size 10
```
