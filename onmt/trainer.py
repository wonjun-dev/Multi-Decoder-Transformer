"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from copy import deepcopy
import torch
import numpy as np
import traceback

import onmt.utils
from onmt.utils.logging import logger
import onmt.utils.latent_utils as latent_utils
import math

import pdb


def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss_A, train_loss_B = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, reduce=False)
    latent_loss = None
    if model.n_latent > 1:
        latent_loss = onmt.utils.loss.build_loss_compute(model,
                                                         tgt_field,
                                                         opt,
                                                         train=False,
                                                         reduce=False)
    valid_loss_A, valid_loss_B = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False, reduce=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model,
                           train_loss_A,
                           train_loss_B,
                           valid_loss_A,
                           train_loss_B,
                           optim,
                           trunc_size,
                           shard_size,
                           norm_method,
                           accum_count,
                           accum_steps,
                           n_gpu,
                           gpu_rank,
                           gpu_verbose_level,
                           report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           latent_loss=latent_loss,
                           segment_token_idx=opt.segment_token_idx)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """
    def __init__(self,
                 model,
                 train_loss_A,
                 train_loss_B,
                 valid_loss_A,
                 valid_loss_B,
                 optim,
                 trunc_size=0,
                 shard_size=32,
                 norm_method="sents",
                 accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1,
                 gpu_rank=1,
                 gpu_verbose_level=0,
                 report_manager=None,
                 model_saver=None,
                 average_decay=0,
                 average_every=1,
                 model_dtype='fp32',
                 earlystopper=None,
                 dropout=[0.3],
                 dropout_steps=[0],
                 latent_loss=None,
                 segment_token_idx=None):
        # Basic attributes.
        self.model = model
        self.n_latent = model.n_latent
        self.train_loss_A = train_loss_A
        self.train_loss_B = train_loss_B
        self.valid_loss_A = valid_loss_A
        self.valid_loss_B = valid_loss_B
        self.latent_loss = latent_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.segment_token_idx = segment_token_idx
        # self.tracking_data = {i: list() for i in range(48)}
        # self.tracking_dec = {0: [], 1: []}
        self.r_t = 1
        self.multi_r_idxs = np.load('multi_r_idxs.npy')
        self.tracking_batch_A = []
        self.tracking_batch_B = []

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d" %
                            (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss_A.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [
                params.detach().float() for params in self.model.parameters()
            ]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics(n_latent=self.n_latent)
        report_stats = onmt.utils.Statistics(n_latent=self.n_latent)
        self._start_report_manager(start_time=total_stats.start_time)

        stats_manager = onmt.utils.StatsManager()

        for i, (batches,
                normalization) in enumerate(self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d" %
                            (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(
                    onmt.utils.distributed.all_gather_list(normalization))

            self._gradient_accumulation(batches, normalization, total_stats,
                                        report_stats, step)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            if step % self.report_manager.report_every == 0:
                stats_manager.add_stats(
                    train_stats={
                        'step': step,
                        'acc': report_stats.accuracy(),
                        'ppl': report_stats.ppl()
                    })

            report_stats = self._maybe_report_training(
                step, train_steps, self.optim.learning_rate(), report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d' %
                                (self.gpu_rank, step))
                valid_stats = self.validate(valid_iter,
                                            moving_average=self.moving_average)
                # print("Current R(T)", self.r_t)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d' %
                                (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step,
                                  valid_stats=valid_stats)

                stats_manager.add_stats(
                    valid_stats={
                        'step': step,
                        'acc': valid_stats.accuracy(),
                        'ppl': valid_stats.ppl()
                    })
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                    and (save_checkpoint_steps != 0
                         and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats, stats_manager, self.tracking_batch_A, self.tracking_batch_B

    def validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics(n_latent=self.n_latent)

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                tgt = batch.tgt

                segment_input = None
                if self.segment_token_idx is not None:
                    segment_input = latent_utils.get_segment_input(
                        tgt, self.segment_token_idx)

                if self.n_latent > 1:
                    seq_len, batch_sz, _ = tgt.size()
                    latent_inputs = latent_utils.get_latent_inputs(
                        self.n_latent, seq_len - 1, batch_sz)

                    all_losses = []
                    for latent_input in latent_inputs:
                        latent_outputs, _ = valid_model(
                            src,
                            tgt,
                            src_lengths,
                            latent_input=latent_input,
                            segment_input=segment_input)
                        latent_losses, _ = self.latent_loss(
                            batch,
                            latent_outputs,
                            None,
                            n_latent=self.n_latent)
                        all_losses.append(latent_losses)
                    all_losses = torch.stack(all_losses, dim=1)
                    max_latent = torch.argmin(all_losses, dim=1)
                    max_latent_input = max_latent.view([1, -1]).repeat(
                        [seq_len - 1, 1])

                    outputs, attns = valid_model(src,
                                                 tgt,
                                                 src_lengths,
                                                 latent_input=max_latent_input,
                                                 segment_input=segment_input)

                    latent_counts = np.zeros([self.n_latent])
                    for i in range(self.n_latent):
                        latent_counts[i] = torch.sum(max_latent == i).item()
                else:
                    # F-prop through the model.
                    outputs_A, outputs_B, attns_A, attns_B, rout_prob = valid_model(
                        src, tgt, src_lengths, segment_input=segment_input)

                # Compute loss.
                loss_A, scores_A, gtruth, max_len, batch_stats_A = self.valid_loss_A(
                    batch, outputs_A, attns_A, n_latent=self.n_latent)
                loss_B, scores_B, gtruth, max_len, batch_stats_B = self.valid_loss_B(
                    batch, outputs_B, attns_B, n_latent=self.n_latent)

                # TODO Binary routing
                binary_routing = (rout_prob > 0.5).float()
                loss = binary_routing[:,
                                      0] * loss_A + binary_routing[:,
                                                                   1] * loss_B
                scores_A_mask = binary_routing[:, 0].repeat_interleave(
                    max_len).unsqueeze(-1)
                scores_B_mask = binary_routing[:, 1].repeat_interleave(
                    max_len).unsqueeze(-1)
                scores_A = scores_A * scores_A_mask
                scores_B = scores_B * scores_B_mask
                scores = scores_A + scores_B
                gtruth_A = gtruth * scores_A_mask.unsqueeze(
                    -1) + gtruth * scores_B_mask.unsqueeze(-1)

                # compare loss and select decoder
                # loss_A, loss_B, score_A, score_B, mask_A, mask_B = self._compare_loss(
                #     loss_A, loss_B, scores_A, scores_B, max_len)
                # loss = (loss_A + loss_B).sum()
                # score = score_A + score_B
                # batch_stats = self.valid_loss_A._stats(loss.clone(), score,
                #                                        gtruth, self.n_latent)

                # select R(T) % instance of each decoder and masking
                # TODO epoch , stats
                # batch_size = tgt.size()[1]
                # mask_A, score_mask_A = self._select_instance(
                #     loss_A, step, batch_size, max_len)
                # mask_B, score_mask_B = self._select_instance(
                #     loss_B, step, batch_size, max_len)
                # loss_A = loss_A * mask_B
                # loss_B = loss_B * mask_A
                # scores_A = scores_A * score_mask_B
                # scores_B = scores_B * score_mask_A

                batch_stats_A = self.valid_loss_A._stats(
                    loss.sum().clone(), scores, gtruth, self.n_latent)
                # batch_stats_B = self.valid_loss_A._stats(
                #     (binary_routing[:, 1] * loss_B).sum().clone(), scores_B,
                #     gtruth, self.n_latent)

                if self.n_latent > 1:
                    batch_stats.latent_counts += latent_counts

                # Update statistics.
                stats.update(batch_stats_A)
                # stats.update(batch_stats_B)

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats, step):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j:j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                segment_input = None
                if self.segment_token_idx is not None:
                    segment_input = latent_utils.get_segment_input(
                        tgt, self.segment_token_idx)

                if self.n_latent > 1:
                    self.model.eval(
                    )  # Turn off dropout when choosing latent classes
                    seq_len, batch_sz, _ = tgt.size()

                    latent_inputs = latent_utils.get_latent_inputs(
                        self.n_latent, seq_len - 1, batch_sz)

                    with torch.no_grad():
                        all_losses = []
                        for latent_input in latent_inputs:
                            latent_outputs, _ = self.model(
                                src,
                                tgt,
                                src_lengths,
                                latent_input=latent_input,
                                segment_input=segment_input)
                            latent_losses, _ = self.latent_loss(
                                batch, latent_outputs, None)
                            all_losses.append(latent_losses)
                        all_losses = torch.stack(all_losses, dim=1)
                        max_latent = torch.argmin(all_losses, dim=1)
                        max_latent_input = max_latent.view([1, -1]).repeat(
                            [seq_len - 1, 1])

                    self.model.train()
                    outputs, attns = self.model(src,
                                                tgt,
                                                src_lengths,
                                                bptt=bptt,
                                                latent_input=max_latent_input,
                                                segment_input=segment_input)

                    latent_counts = np.zeros([self.n_latent])
                    for i in range(self.n_latent):
                        latent_counts[i] = torch.sum(max_latent == i).item()
                else:
                    outputs_A, outputs_B, attns_A, attns_B, rout_prob = self.model(
                        src,
                        tgt,
                        src_lengths,
                        bptt=bptt,
                        segment_input=segment_input)

                bptt = True

                # 3. Compute loss.
                # try:
                # loss for decoder_A
                # loss_A: (bs)
                # scores_A: (bs*max_len, vocab)
                loss_A, scores_A, gtruth, max_len, batch_stats_A = self.train_loss_A(
                    batch,
                    outputs_A,
                    attns_A,
                    normalization=normalization,
                    shard_size=self.shard_size,
                    trunc_start=j,
                    trunc_size=trunc_size,
                    n_latent=self.n_latent)
                # loss for decoder_B
                loss_B, scores_B, gtruth, max_len, batch_stats_B = self.train_loss_B(
                    batch,
                    outputs_B,
                    attns_B,
                    normalization=normalization,
                    shard_size=self.shard_size,
                    trunc_start=j,
                    trunc_size=trunc_size,
                    n_latent=self.n_latent)

                reg = (loss_A - loss_B)**2
                loss = rout_prob[:,
                                 0] * loss_A + rout_prob[:,
                                                         1] * loss_B + 0.1 * reg

                # # select R(T) % instance of each decoder and masking
                # # TODO epoch , stats
                # batch_size = tgt.size()[1]
                # mask_A, score_mask_A = self._select_instance(
                #     loss_A, step, batch_size, max_len)
                # mask_B, score_mask_B = self._select_instance(
                #     loss_B, step, batch_size, max_len)
                # loss_A = loss_A * mask_B
                # loss_B = loss_B * mask_A
                # scores_A = scores_A * score_mask_B
                # scores_B = scores_B * score_mask_A
                # print(scores_A.shape)  # (max_len * bs, vocab)
                # print(gtruth.shape)  # (max_len * bs)

                batch_stats_A = self.train_loss_A._stats(
                    (rout_prob[:, 0] * loss_A).sum().clone(), scores_A, gtruth,
                    self.n_latent)
                batch_stats_B = self.train_loss_A._stats(
                    (rout_prob[:, 1] * loss_B).sum().clone(), scores_B, gtruth,
                    self.n_latent)

                # # # tracking
                # # self._tracking(batch.indices, mask_A, mask_B)
                # self._tracking(batch.indices, mask_A, mode='A')
                # self._tracking(batch.indices, mask_B, mode='B')
                if self.n_latent > 1:
                    batch_stats.latent_counts += latent_counts

                total_stats.update(batch_stats_A)
                report_stats.update(batch_stats_A)
                total_stats.update(batch_stats_B)
                report_stats.update(batch_stats_B)

                # # TODO aggregate losses
                # divider = mask_A + mask_B
                # zero_idx = torch.where(divider == 0)
                # divider[zero_idx] = 1
                # loss = ((loss_A + loss_B) / divider).sum()
                loss = loss.sum()
                loss /= float(normalization)
                if loss is not None:
                    self.optim.backward(loss)

                # except Exception:
                #     traceback.print_exc()
                #     logger.info("At step %d, we removed a batch - accum %d",
                #                 self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [
                            p.grad.data for p in self.model.parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder_A.state is not None:
                    self.model.decoder_A.detach_state()
                if self.model.decoder_B.state is not None:
                    self.model.decoder_B.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(step,
                                                       num_steps,
                                                       learning_rate,
                                                       report_stats,
                                                       multigpu=self.n_gpu > 1)

    def _report_step(self,
                     learning_rate,
                     step,
                     train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(learning_rate,
                                                   step,
                                                   train_stats=train_stats,
                                                   valid_stats=valid_stats)

    def _compare_loss(self, loss_A, loss_B, scores_A, scores_B, max_len):
        """
        Compare loss_A with loss_B and masking bigger ones.
        """
        mask_A = (loss_A < loss_B).float()
        mask_B = (loss_B < loss_A).float()
        score_mask_A = mask_A.repeat_interleave(max_len).unsqueeze(-1)
        score_mask_B = mask_B.repeat_interleave(max_len).unsqueeze(-1)
        loss_A = loss_A * mask_A
        loss_B = loss_B * mask_B
        scores_A = scores_A * score_mask_A
        scores_B = scores_B * score_mask_B

        return loss_A, loss_B, scores_A, scores_B, mask_A, mask_B

    # def _tracking(self, indices, mask_A, mask_B):
    #     idx_stat = self.tracking_data
    #     dec_stat = self.tracking_dec
    #     for i, batch_idx in enumerate(indices):
    #         if batch_idx.item() in idx_stat.keys():
    #             idx_stat[batch_idx.item()].append(int(mask_A[i].item()))
    #         else:
    #             pass

    #     dec_stat[0].append(mask_A.sum().item())
    #     dec_stat[1].append(mask_B.sum().item())

    def _select_instance(self, loss, step, batch_size, max_len):
        epoch_step = 4300
        epoch = math.floor(step / 4300)
        tau = 0.7
        t_k = 10
        self.r_t = 1 - tau * min(epoch / t_k, 1)  # TODO epoch define
        instance_num = int(self.r_t * batch_size)
        large_idx = torch.topk(loss, batch_size - instance_num).indices.cuda()
        mask = torch.ones_like(loss)
        mask[large_idx] = 0
        score_mask = mask.repeat_interleave(max_len).unsqueeze(-1)

        return mask, score_mask

    def _tracking(self, indices, mask, mode='A'):
        selected_idxs = torch.where(mask != 0)
        selected_idxs = indices[selected_idxs]
        selected_size = len(selected_idxs)
        cnt = 0
        for i, batch_idx in enumerate(selected_idxs):
            if batch_idx.item() in self.multi_r_idxs:
                cnt += 1
            else:
                pass
        ratio = cnt / selected_size
        if mode == 'A':
            self.tracking_batch_A.append(ratio)
        else:
            self.tracking_batch_B.append(ratio)
