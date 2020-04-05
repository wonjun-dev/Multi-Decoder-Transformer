import GPUtil
import argparse
import pandas as pd
import numpy as np
import os
import csv

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-target_dir',
                    help='Target directory contains results of each seed')

GPUs = GPUtil.getGPUs()
GPUavailability = GPUtil.getAvailability(GPUs,
                                         maxLoad=1,
                                         maxMemory=1,
                                         includeNan=False,
                                         excludeID=[],
                                         excludeUUID=[])

finish_exp = sum(GPUavailability) == len(GPUavailability)


def main(args):
    target_dir = args.target_dir
    if finish_exp:
        seeds = ['2020', '2021', '2022', '2023']
        acc = np.zeros(50)
        invalid = np.zeros(50)
        repeat = np.zeros(50)
        for n, seed in enumerate(seeds):
            file = target_dir + seed + '.csv'
            stats = pd.read_csv(file, names=['acc', 'invalid', 'repeat'])
            for i in range(len(stats)):
                acc[i] += stats.loc[i, 'acc']
                invalid[i] += stats.loc[i, 'invalid']
                repeat[i] += stats.loc[i, 'repeat']

        acc = acc / (n + 1)
        invalid = invalid / (n + 1)
        repeat = repeat / (n + 1)

        f = open(target_dir + 'avg.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        for i in range(50):
            wr.writerow([acc[i], invalid[i], repeat[i]])
        f.close()
    else:
        pass


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)