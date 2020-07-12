import numpy as np


class StatsManager(object):
    def __init__(self, stat_names=['step', 'acc', 'ppl', 'auc', 'xent']):
        self.stat_names = stat_names
        self.train_stats = {}
        self.val_stats = {}
        self.test_stats = {}

        for name in stat_names:
            self.train_stats[name] = []
            self.val_stats[name] = []
            self.test_stats[name] = []

    def add_stats(self, train_stats=None, valid_stats=None, test_stats=None):
        assert train_stats is not None or valid_stats is not None or test_stats is not None

        if train_stats is not None:
            for name, val in train_stats.items():
                self.train_stats[name].append(val)
            return

        if valid_stats is not None:
            for name, val in valid_stats.items():
                self.val_stats[name].append(val)
            return

        if test_stats is not None:
            for name, val in test_stats.items():
                self.test_stats[name].append(val)

    def get_best_model(self, stat_name='acc', highest_best=True):
        stat_list = np.array(self.val_stats[stat_name])[:]
        print('stat_list', stat_list)

        if highest_best:
            best_idx = np.argmax(stat_list)
        else:
            best_idx = np.argmin(stat_list)

        best_stats = {}
        for name in self.stat_names:
            best_stats[name] = self.val_stats[name][:][best_idx]

        return self.val_stats['step'][:][best_idx], best_stats

    def write_stats(self, output_dir):
        with open('%s/train_stats.csv' % output_dir, 'w+') as train_file:
            steps = self.train_stats['step']
            for idx, step in enumerate(steps):
                acc = self.train_stats['acc'][idx] if 'acc' in self.train_stats else 0
                auc = self.train_stats['auc'][idx] if 'auc' in self.train_stats else 0
                ppl = self.train_stats['ppl'][idx] if 'ppl' in self.train_stats else 0
                xent = self.train_stats['xent'][idx] if 'xent' in self.train_stats else 0

                train_file.write('{},{},{},{},{}\n'.format(step, acc, xent, auc, ppl))

        with open('%s/valid_stats.csv' % output_dir, 'w+') as valid_file:
            steps = self.val_stats['step']
            for idx, step in enumerate(steps):
                acc = self.val_stats['acc'][idx] if 'acc' in self.val_stats else 0
                auc = self.val_stats['auc'][idx] if 'auc' in self.val_stats else 0
                ppl = self.val_stats['ppl'][idx] if 'ppl' in self.val_stats else 0
                xent = self.val_stats['xent'][idx] if 'xent' in self.val_stats else 0

                valid_file.write('{},{},{},{},{}\n'.format(step, acc, xent, auc, ppl))

        with open('%s/test_stats.csv' % output_dir, 'w+') as test_file:
            steps = self.test_stats['step']
            for idx, step in enumerate(steps):
                acc = self.test_stats['acc'][idx] if 'acc' in self.test_stats else 0
                auc = self.test_stats['auc'][idx] if 'auc' in self.test_stats else 0
                ppl = self.test_stats['ppl'][idx] if 'ppl' in self.test_stats else 0
                xent = self.test_stats['xent'][idx] if 'xent' in self.test_stats else 0

                test_file.write('{},{},{},{},{}\n'.format(step, acc, xent, auc, ppl))
