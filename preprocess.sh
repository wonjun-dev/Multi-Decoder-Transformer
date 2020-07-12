dataset=chembl_lm #USPTO-50k_no_rxn_aug_2
#dataset=hiv
#python preprocess.py -train_src  data/hiv/smiles.txt \
#                     -train_tgt data/hiv/smiles.txt \
#                     -save_data data/${dataset}/${dataset} \
#                     -src_seq_length 1000 -tgt_seq_length 1000 \
#                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
#                     -overwrite -add_mask_token \
#dataset=USPTO-50k_no_rxn
#python preprocess.py -train_src data/${dataset}/src-train.txt \
#                    -train_tgt data/${dataset}/tgt-train.txt \
#                    -valid_src data/${dataset}/src-val.txt \
#                    -valid_tgt data/${dataset}/tgt-val.txt \
#                    -save_data data/${dataset}/${dataset} \
#                     -src_seq_length 1000 -tgt_seq_length 1000 \
#                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
#                     -n_smiles_aug 1 \
#                     -overwrite \
#                     -src_vocab data/chembl_lm/chembl_lm-merge.vocab.pt \
python preprocess_merge.py -train_src data/${dataset}/src-train.txt \
                                data/hiv/smiles.txt \
                                data/USPTO-50k_no_rxn/src-train.txt \
                                data/pcba/smiles.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                                data/hiv/smiles.txt \
                                data/USPTO-50k_no_rxn/tgt-train.txt \
                                data/pcba/smiles.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset}-merge2 \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab \
                     -contrastive -n_smiles_aug 1 \
                     -get_adj \
                     -get_attr -get_new2canon \
                     -overwrite -add_mask_token \
                     #-src_vocab data/chembl_lm/chembl_lm.vocab.pt \
                     #-src_vocab data/USPTO-50k_no_rxn_aug_2/USPTO-50k_no_rxn_aug_2.vocab.pt
                     # -random_mask \
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
