dataset=hiv #USPTO-50k_no_rxn_aug_2
seed_list=(2020 2021 2022 2023)

for seed in ${seed_list[@]}; do
    python preprocess_xy.py -train_src data/${dataset}/src-train-${seed}.txt \
                     -train_tgt data/${dataset}/tgt-train-${seed}.npy \
                     -valid_src data/${dataset}/src-val-${seed}.txt \
                     -valid_tgt data/${dataset}/tgt-val-${seed}.npy \
                     -save_data data/${dataset}/${dataset}-${seed} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
                     -src_vocab data/chembl_lm/chembl_lm.vocab.pt \
                     -overwrite -contrastive #-add_mask_token \
                     #-contrastive -n_smiles_aug 1 \
                     #-src_vocab data/USPTO-50k_no_rxn_aug_2/USPTO-50k_no_rxn_aug_2.vocab.pt
                     # -random_mask \
done
#python preprocess_tgtaug.py -train_src data/${dataset}/src-train.txt \
#                     -train_tgt1 data/${dataset}/tgt1-train.txt \
#                     -train_tgt2 data/${dataset}/tgt2-train.txt \
#                     -valid_src data/${dataset}/src-val.txt \
#                     -valid_tgt1 data/${dataset}/tgt1-val.txt \
#                     -valid_tgt2 data/${dataset}/tgt2-val.txt \
#                     -save_data data/${dataset}/${dataset} \
#                     -src_seq_length 1000 -tgt_seq_length 1000 \
#                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
#                     -overwrite
#python preprocess.py -train_src data/${dataset}/srcjakaug-train.txt \
#                     -train_tgt data/${dataset}/tgtjakaug-train.txt \
#                     -valid_src data/${dataset}/srcjakaug-val.txt \
#                     -valid_tgt data/${dataset}/tgtjakaug-val.txt \
#                     -save_data data/${dataset}/${dataset}-jakaug \
#                     -src_seq_length 1000 -tgt_seq_length 1000 \
#                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
#                     -overwrite
#python preprocess.py -train_src data/${dataset}/src-train-new-beam-${beam}.txt \
#                     -train_tgt data/${dataset}/tgt-train-new-beam-${beam}.txt \
#                     -valid_src data/${dataset}/src-val-new-beam-${beam}.txt \
#                     -valid_tgt data/${dataset}/tgt-val-new-beam-${beam}.txt \
#                     -save_data data/${dataset}/${dataset}-new-beam-${beam} \
#                     -src_seq_length 1000 -tgt_seq_length 1000 \
#                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
#                     -overwrite
#python preprocess.py -train_src data/${dataset}/src-train-hint-${beam}.txt \
#                     -train_tgt data/${dataset}/tgt-train-hint-${beam}.txt \
#                     -valid_src data/${dataset}/src-val-hint-${beam}.txt \
#                     -valid_tgt data/${dataset}/tgt-val-hint-${beam}.txt \
#                     -save_data data/${dataset}/${dataset}-hint-${beam} \
#                     -src_seq_length 1000 -tgt_seq_length 1000 \
#                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
#                     -overwrite

#beam=10
#python preprocess.py -train_src data/${dataset}/src-train-woprod-${beam}.txt \
#                     -train_tgt data/${dataset}/tgt-train-new-beam-${beam}.txt \
#                     -valid_src data/${dataset}/src-val-woprod-${beam}.txt \
#                     -valid_tgt data/${dataset}/tgt-val-new-beam-${beam}.txt \
#                     -save_data data/${dataset}/${dataset}-woprod-${beam} \
#                     -src_seq_length 1000 -tgt_seq_length 1000 \
#                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
#                     -overwrite
