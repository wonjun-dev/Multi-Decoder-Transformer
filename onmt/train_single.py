#!/usr/bin/env python
"""Training on a single process."""
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from tqdm import tqdm

import pdb
from augment import augment_smiles
from itertools import combinations


def _create_output_dirs(opt):
    if not os.path.exists(opt.save_model):
        os.makedirs(opt.save_model)
    models_dir = opt.save_model + '/models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    models_path = models_dir + '/model'
    opt.save_model = models_path


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def write_opt(file_path, opt):
    file = open(file_path, 'w+')

    for name, val in vars(opt).items():
        file.write('%s: %s\n' % (name, str(val)))


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(vocab,
                                opt.model_type,
                                dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    vocab_dict = vocab['src'].base_field.vocab.stoi
    print(vocab_dict)
    # input()
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    output_dir = opt.save_model
    _create_output_dirs(opt)

    write_opt('%s/opts.txt' % (output_dir), opt)

    # Build optimizer.
    # optimizer = Optimizer.from_opt(model, opt, checkpoint=checkpoint)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Build model saver
    # model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    # trainer = build_trainer(opt,
    #                         device_id,
    #                         model,
    #                         fields,
    #                         optim,
    #                         model_saver=model_saver)

    if batch_queue is None:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt)

    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                yield batch

        train_iter = _train_iter()

    # valid_iter = build_dataset_iter("valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    model.train()
    for batch in train_iter:
        optimizer.zero_grad()
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
        aug_src, aug_src_lengths = augment_smiles(batch, vocab_dict)

        for i in range(2):
            if i == 0:
                z_i = model(src, src_lengths)
            else:
                z_j = model(aug_src, aug_src_lengths)

        # TODO: Contrastive Loss
        batch_size = z_i.size(0)
        v_dict = {}
        for k in range(batch_size):
            v_dict[2 * k] = z_i[k].unsqueeze(0)
            v_dict[2 * k + 1] = z_j[k].unsqueeze(0)

        sim_arr = torch.zeros(2 * batch_size, 2 * batch_size)
        for c in combinations([i for i in range(batch_size * 2)], 2):
            sim = F.cosine_similarity(v_dict[c[0]], v_dict[c[1]])
            sim_arr[c[0], c[1]] = sim
            sim_arr[c[1], c[0]] = sim
        # define loss
        loss = 0
        for k in range(batch_size):
            denominator1 = sum(torch.exp(sim_arr[2 * k]))
            denominator2 = sum(torch.exp(sim_arr[2 * k + 1]))
            numerator1 = torch.exp(sim_arr[2 * k][2 * k + 1])
            numerator2 = torch.exp(sim_arr[2 * k + 1][2 * k])
            loss1 = -torch.log(numerator1 / denominator1)
            loss2 = -torch.log(numerator2 / denominator2)
            loss += loss1 + loss2
        loss = loss / (2 * batch_size)
        loss.backward()
        print(loss.item())
        optimizer.step()

        torch.save(model.state_dict(), 'encoder.pt')

    # inter_sims = []

    # for i in range(batch_size):
    #     z_i_ = z_i[i].unsqueeze(0)
    #     for j in range(batch_size):
    #         z_j_ = z_j[j].unsqueeze(0)
    #         sim = F.cosine_similarity(z_i_, z_j_)
    #         inter_sims.append(sim)
    # intra_sims1 = []
    # intra_sims2 = []
    # for i in range(batch_size):
    #     z_i_ = z_i[i].unsqueeze(0)
    #     z_j_ = z_j[i].unsqueeze(0)
    #     for j in range(i + 1, batch_size):
    #         z_i__ = z_i[j].unsqueeze(0)
    #         z_j__ = z_j[j].unsqueeze(0)
    #         sim1 = F.cosine_similarity(z_i_, z_i__)
    #         sim2 = F.cosine_similarity(z_j_, z_j__)
    #         intra_sims1.append(sim1)
    #         intra_sims2.append(sim2)

    # for k in range(batch_size):
    #     denominator = torch.exp(
    #         sum(sims[k * batch_size:(k + 1) * batch_size]))
    #     numerator = torch.exp(sims[k * batch_size])
    #     loss = -torch.log(numerator / denominator)

    #     print('loss', loss)
    #     input()

    # total_stats, stats_manager = trainer.train(
    #     train_iter,
    #     train_steps,
    #     save_checkpoint_steps=opt.save_checkpoint_steps,
    #     valid_iter=valid_iter,
    #     valid_steps=opt.valid_steps)

    # stats_manager.write_stats(output_dir=output_dir)

    # best_model_step, best_model_stats = stats_manager.get_best_model()
    # best_model_path = opt.save_model + '_step_%d.pt' % best_model_step

    # with open('%s/summary.txt' % output_dir, 'w+') as summary_file:
    #     summary_file.write('best_model_path: %s\n' % best_model_path)
    #     for name, val in best_model_stats.items():
    #         summary_file.write('%s,%s\n' % (name, str(val)))

    # if opt.tensorboard:
    #     trainer.report_manager.tensorboard_writer.close()

    # return best_model_path
