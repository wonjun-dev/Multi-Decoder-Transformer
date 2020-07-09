import rdkit.Chem as Chem
import numpy as np
import re
from collections import defaultdict
import copy
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


def find_atomic_token(tokenized):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)" #|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    is_atomic = [regex.match(token) is not None for token in tokenized]
    return np.array(is_atomic)

#max_val = 0
#min_val = 100
def adj2khop(adj, K):
    n = adj.shape[0]
    A = adj
    C = copy.deepcopy(A)
    for k in range(2, K+1):
        B = np.zeros_like(A)
        tmp = np.matmul(A, C)
        B[(tmp > 0) & (C == 0)] = k
        B[np.arange(n), np.arange(n)] = 0
        C = C + B
    return C


def get_mol_and_adj(smiles, K=4):
    global max_val
    global min_val
    mol = Chem.MolFromSmiles(smiles)
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    adj[adj==1.5] = 4
    khop = adj2khop((adj>0).astype(int), K)
    atoms = mol.GetAtoms()
    features = dict({})
    features['valence'] = np.array([atom.GetExplicitValence() for atom in
        atoms]) -1
    features['degree'] = np.array([atom.GetDegree() for atom in atoms])
    features['atomic'] = np.array([atom.GetAtomicNum() for atom in atoms]) -1
    features['is_aromatic'] = np.array([int(atom.GetIsAromatic()) for atom in atoms])
    features['num_h'] = np.array([atom.GetTotalNumHs() for atom in atoms])
    features['formal_charge'] = np.array([atom.GetFormalCharge() for atom in
        atoms]) + 5
    #print(features['valence'])
    #max_val = max(max_val, features['valence'].max())
    #min_val = min(min_val, features['valence'].min())
    #print(max_val, min_val)
    return mol, adj, khop, features

def enum_smiles_from_mol(mol):
    #mol = Chem.MolFromSmiles(canon_smiles)
    ans = list(range(mol.GetNumAtoms()))
    np.random.shuffle(ans)
    new_mol = Chem.RenumberAtoms(mol, ans)
    new_smiles = Chem.MolToSmiles(new_mol, canonical=False)
    smiles_order = np.array(new_mol.GetPropsAsDict(True, True)['_smilesAtomOutputOrder'])
    ans = np.array(ans)
    new2canon = ans[smiles_order]
    return new_smiles, new2canon


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


def read_file(file_path, beam_size=1, max_read=-1, parse_func=None):
    read_file = open(file_path, 'r+')
    output_list = []  # List of beams if beam_size is > 1 else list of smiles
    cur_beam = []  # Keep track of the current beam

    for line in read_file.readlines():
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
