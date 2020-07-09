""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0, score=[],
            target=[], n_latent=1):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.score = score
        self.target = target
        self.start_time = time.time()

        self.latent_counts = np.zeros([n_latent])

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss = self.loss + stat.loss
        self.n_words =  self.n_words + stat.n_words
        self.n_correct = self.n_correct + stat.n_correct
        self.score = self.score + stat.score
        self.target = self.target + stat.target

        if update_n_src_words:
            self.n_src_words = self.n_src_words + stat.n_src_words

        self.latent_counts = self.latent_counts + stat.latent_counts

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def auc(self):
        if len(self.score) > 0 and len(self.target):
            score = np.concatenate(self.score, axis=0)
            target = np.concatenate(self.target, axis=0)
            return roc_auc_score(target, score)
        else:
            return 0

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return np.exp(np.min((self.loss / self.n_words, 100 *
            np.ones_like(self.n_words)), axis=0))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def latent_output(self):
        if len(self.latent_counts) == 1:
            return ''
        latent_str = 'Latent:'
        sum_counts = float(np.sum(self.latent_counts))
        for i in range(len(self.latent_counts)):
            latent_str += '%d:%.2f,' % (i, self.latent_counts[i] / sum_counts)
        return latent_str

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
                ("Step {}; acc: {}; ppl: {}; xent: {} auc: {}; " +
                "lr: {:.2f}; {} {:.2f}/{} tok/s; {:.2f} sec").format(step_fmt,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               self.auc(),
               learning_rate,
               self.latent_output(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/auc", self.auc(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
