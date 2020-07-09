#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import sys
import gc
import torch
from functools import partial
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab,\
                                    _load_vocab

import pdb


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def build_save_dataset(corpus_type, fields, src_reader, tgt1_reader,
                       tgt2_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        # TODO : tgt1, tgt2, 근데 tgt1 이랑 tgt2가 어차피 같은애라서 굳이?
        counters = defaultdict(Counter)
        srcs = opt.train_src
        tgts1 = opt.train_tgt1
        tgts2 = opt.train_tgt2
        ids = opt.train_ids
    else:
        srcs = [opt.valid_src]
        tgts1 = [opt.valid_tgt1]
        tgts2 = [opt.valid_tgt2]
        ids = [None]

    for src, tgt1, tgt2, maybe_id in zip(srcs, tgts1, tgts2, ids):
        logger.info("Reading source and target files: %s %s %s." %
                    (src, tgt1, tgt2))
        # TODO tgt1_shards, tgt2_shards
        src_shards = split_corpus(src, opt.shard_size)
        tgt1_shards = split_corpus(tgt1, opt.shard_size)
        tgt2_shards = split_corpus(tgt2, opt.shard_size)
        shard_pairs = zip(src_shards, tgt1_shards, tgt2_shards)
        dataset_paths = []
        if (corpus_type == "train"
                or opt.filter_valid) and tgt1 is not None and tgt2 is not None:
            filter_pred = partial(inputters.filter_example,
                                  use_src_len=opt.data_type == "text",
                                  max_src_len=opt.src_seq_length,
                                  max_tgt_len=opt.tgt_seq_length)
        else:
            filter_pred = None

        if corpus_type == "train":
            existing_fields = None
            if opt.src_vocab != "":
                try:
                    logger.info("Using existing vocabulary...")
                    existing_fields = torch.load(opt.src_vocab)
                except torch.serialization.pickle.UnpicklingError:
                    logger.info("Building vocab from text file...")
                    src_vocab, src_vocab_size = _load_vocab(
                        opt.src_vocab, "src", counters,
                        opt.src_words_min_frequency)
            else:
                src_vocab = None
            if opt.tgt_vocab != "":
                tgt_vocab, tgt_vocab_size = _load_vocab(
                    opt.tgt_vocab, "tgt", counters,
                    opt.tgt_words_min_frequency)
            else:
                tgt_vocab = None

        for i, (src_shard, tgt1_shard, tgt2_shard) in enumerate(shard_pairs):
            assert len(src_shard) == len(tgt1_shard) == len(tgt2_shard)
            logger.info("Building shard %d." % i)
            dataset = inputters.Dataset(
                fields,
                readers=([src_reader, tgt1_reader, tgt2_reader]
                         if tgt1_reader else [src_reader]),
                data=([("src", src_shard), ("tgt1", tgt1_shard),
                       ("tgt2",
                        tgt2_shard)] if tgt1_reader else [("src", src_shard)]),
                dirs=([opt.src_dir, None, None]
                      if tgt1_reader else [opt.src_dir]),
                sort_key=inputters.str2sortkey[opt.data_type],
                filter_pred=filter_pred)
            if corpus_type == "train" and existing_fields is None:
                for ex in dataset.examples:
                    for name, field in fields.items():
                        try:
                            f_iter = iter(field)
                        except TypeError:
                            f_iter = [(name, field)]
                            all_data = [getattr(ex, name, None)]
                        else:
                            all_data = getattr(ex, name)
                        for (sub_n, sub_f), fd in zip(f_iter, all_data):
                            has_vocab = (sub_n == 'src' and src_vocab) or \
                                        (sub_n == 'tgt' and tgt_vocab)
                            if (hasattr(sub_f, 'sequential')
                                    and sub_f.sequential and not has_vocab):
                                val = fd
                                counters[sub_n].update(val)
            if maybe_id:
                shard_base = corpus_type + "_" + maybe_id
            else:
                shard_base = corpus_type
            data_path = "{:s}.{:s}.{:d}.pt".\
                format(opt.save_data, shard_base, i)
            dataset_paths.append(data_path)

            logger.info(" * saving %sth %s data shard to %s." %
                        (i, shard_base, data_path))

            dataset.save(data_path)

            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    if corpus_type == "train":
        if opt.add_mask_token:
            counters['src']['<MASK>'] = 1
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = _build_fields_vocab(
                fields, counters, opt.data_type, opt.share_vocab,
                opt.vocab_size_multiple, opt.src_vocab_size,
                opt.src_words_min_frequency, opt.tgt_vocab_size,
                opt.tgt_words_min_frequency)
        else:
            fields = existing_fields
        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(train_dataset,
                                   fields,
                                   opt.data_type,
                                   opt.share_vocab,
                                   opt.src_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency,
                                   vocab_size_multiple=opt.vocab_size_multiple)
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    if not (opt.overwrite):
        check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = 0
    tgt1_nfeats = 0
    tgt2_nfeats = 0
    # TODO train_tgt_2 만들어서 zip -> tgt1, tgt2
    for src, tgt1, tgt2 in zip(opt.train_src, opt.train_tgt1, opt.train_tgt2):
        src_nfeats += count_features(src) if opt.data_type == 'text' \
            else 0
        tgt1_nfeats += count_features(tgt1)
        tgt2_nfeats += count_features(tgt2)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt1_nfeats)
    logger.info(" * number of target features: %d." % tgt2_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type,
                                  src_nfeats,
                                  tgt1_nfeats,
                                  tgt2_nfeats,
                                  dynamic_dict=opt.dynamic_dict,
                                  src_truncate=opt.src_seq_length_trunc,
                                  tgt_truncate=opt.tgt_seq_length_trunc)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt1_reader = inputters.str2reader["text"].from_opt(opt)
    tgt2_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset('train', fields, src_reader, tgt1_reader, tgt2_reader,
                       opt)

    if opt.valid_src and opt.valid_tgt1 and opt.valid_tgt2:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt1_reader,
                           tgt2_reader, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
