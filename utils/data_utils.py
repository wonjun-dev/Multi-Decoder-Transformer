import rdkit.Chem as Chem
import re
from collections import defaultdict
from operator import itemgetter
import copy
from collections import Counter
import numpy as np
from tqdm import tqdm

import pdb


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None

    if mol is None:
        return ''
    else:
        return Chem.MolToSmiles(mol)


def match_smiles_set(source_set, target_set):
    if len(source_set) != len(target_set):
        return False

    for smiles in target_set:
        if smiles not in source_set:
            return False
    return True


def canonicalize(smiles=None, smiles_list=None):
    """Return the canonicalized version of the given smiles or smiles list"""
    assert (smiles is None) != (smiles_list is None)  # Only take one input

    if smiles is not None:
        return canonicalize_smiles(smiles)
    elif smiles_list is not None:
        # Convert smiles to mol and back to cannonicalize
        new_smiles_list = []

        for smiles in smiles_list:
            new_smiles_list.append(canonicalize_smiles(smiles))
        return new_smiles_list


def parse_rxn_token(token):
    m = re.search('[0-9]+', token)
    rxn_class = int(m[0])
    return rxn_class


def read_src_tgt_files(data_dir,
                       data_type,
                       beam_size=1,
                       max_read=-1,
                       source_func=None,
                       target_func=None):
    source_data = read_file('%s/src-%s.txt' % (data_dir, data_type), beam_size,
                            max_read, source_func)
    target_data = read_file('%s/tgt-%s.txt' % (data_dir, data_type), beam_size,
                            max_read, target_func)
    return source_data, target_data


def read_file(file_path,
              beam_size=1,
              max_read=-1,
              parse_func=None,
              test_ensem=False):
    read_file = open(file_path, 'r+')
    output_list = []  # List of beams if beam_size is > 1 else list of smiles
    cur_beam = []  # Keep track of the current beam

    # TODO  sorting
    lines = read_file.readlines()
    if test_ensem:
        new_lines = copy.deepcopy(lines)
        unit = beam_size  # beam_size should be aug_num * beam_size
        num_unit = int(len(lines) / unit)
        for u in tqdm(range(num_unit)):
            pred = lines[u * unit:(u + 1) * unit]

            # sort by count & beam weight
            smis = []
            probs = []
            canonicalized_smis = []
            for p in pred:
                smi, prob = p.split(',')
                smis.append(smi)
                probs.append(float(prob))
                canonicalized_smis.append(
                    canonicalize(smi.strip().replace(' ', '')))

            # calculate weight
            weights = []
            aug = 20
            bs = 50
            scale = np.array([
                float(i + 1) for _ in range(aug) for i in range(bs)
            ])  # TODO  get argument
            scale = scale * 0.001 + 1
            for smi in canonicalized_smis:
                delta = np.array([float(s == smi) for s in canonicalized_smis])
                delta /= scale
                weights.append(sum(delta))
            smis = np.array(smis)
            probs = np.array(probs)
            weights = np.array(weights)
            canonicalized_smis = np.array(canonicalized_smis)

            sort_by_cnt = Counter(smis).most_common()

            tmp = []
            for item in sort_by_cnt:
                item = list(item)
                smi = item[0]
                cnt = item[1]

                idxs = np.where(canonicalized_smis == canonicalize(
                    smi.strip().replace(' ', '')))[0]
                smi_weight = weights[idxs][0]
                item.append(smi_weight)
                tmp.append(item)
            # sort by cnt first, and sort by log probs sencondly
            tmp = sorted(tmp, key=lambda x: (-x[1], -x[-1]))

            l = []
            for item in tmp:
                cnt = item[1]
                for _ in range(cnt):
                    l.append(item[0] + ',' + str(item[1]) + ',' + str(item[2]))

            # sort by count & log probs
            # smis = []
            # probs = []
            # for p in pred:
            #     smi, prob = p.split(',')
            #     smis.append(smi)
            #     probs.append(float(prob))
            # smis = np.array(smis)
            # probs = np.array(probs)

            # sort_by_cnt = Counter(smis).most_common()

            # tmp = []
            # for item in sort_by_cnt:
            #     item = list(item)
            #     smi = item[0]
            #     cnt = item[1]
            #     idxs = np.where(smis == smi)[0]
            #     smi_probs = probs[idxs]
            #     best_prob = max(smi_probs)
            #     item.append(best_prob)
            #     tmp.append(item)
            # # sort by cnt first, and sort by log probs sencondly
            # tmp = sorted(tmp, key=lambda x: (-x[1], -x[-1]))

            # l = []
            # for item in tmp:
            #     cnt = item[1]
            #     for _ in range(cnt):
            #         l.append(item[0] + ',' + str(item[1]) + ',' + str(item[2]))

            # sort by log probability
            # l = []
            # for p in pred:
            #     p_ = p.split(',')
            #     t = [p_[0], float(p_[1])]
            #     l.append(t)
            # l.sort(key=itemgetter(1), reverse=True)
            # for n, item in enumerate(l):
            #     item[1] = str(item[1])
            #     item = ','.join(item)
            #     l[n] = item

            new_lines[u * unit:(u + 1) * unit] = l

        for idx, row in enumerate(new_lines):
            if idx == 0:
                with open(file_path + '_sorted_cnt_weight.txt', 'w') as f:
                    f.write(row + '\n')
            else:
                with open(file_path + '_sorted_cnt_weight.txt', 'a') as f:
                    f.write(row + '\n')

        lines = new_lines
    for line in lines:
        if parse_func is None:
            parse = line.strip().replace(' ', '')  # default parse function
            if ',' in parse:
                # If output separated by commas, return first by default
                parse = parse.split(',')[0]
        else:
            parse = parse_func(line)

        cur_beam.append(parse)
        if len(cur_beam) == beam_size:
            if beam_size == 1:
                output_list.append(cur_beam[0])
            else:
                output_list.append(cur_beam)
            if max_read != -1 and len(output_list) >= max_read:
                break
            cur_beam = []
    read_file.close()
    return output_list


class StatsTracker():
    def __init__(self):
        self.stats_sum = defaultdict(float)
        self.stats_norm = defaultdict(float)

    def add_stat(self, stat_name, val, norm=1):
        self.stats_sum[stat_name] += val
        self.stats_norm[stat_name] += norm

    def get_stats(self):
        stats = {
            name: self.stats_sum[name] / self.stats_norm[name]
            for name in self.stats_sum
        }
        return stats

    def print_stats(self, pre=''):
        stats = self.get_stats()
        stats_string = ''
        for name, val in stats.items():
            stats_string += '%s: %s ' % (name, str(val))
        stats_string = stats_string[:-1]
        print(pre + ' ' + stats_string)
