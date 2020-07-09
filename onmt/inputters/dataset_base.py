# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab
import random
import numpy as np
import copy
from utils.data_utils import enum_smiles_from_mol, \
        smi_tokenizer, find_atomic_token, get_mol_and_adj
from dgl.data import chem


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example

atom_featurizer = chem.BaseAtomFeaturizer({
            'atom': chem.atom_type_one_hot,
            'degree': chem.atom_total_degree,
            'valence': chem.atom_implicit_valence,
            'formal_charge': chem.atom_formal_charge,
            #'chiral': chem.atom_chiral_tag_one_hot,
            'num_h': chem.atom_total_num_H,
            #'hybridization': atom_hybridization_one_hot,
            'aromatic': chem.atom_is_aromatic
        })


class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, fields, readers, data, dirs, sort_key,
                 filter_pred=None, random_mask=False, contrastive=False,
                 n_smiles_aug=1, get_adj=False, get_attr=False,
                 get_new2canon=False):
        self.sort_key = sort_key
        self.random_mask = random_mask
        self.contrastive = contrastive
        self.n_smiles_aug = n_smiles_aug
        self.get_adj = get_adj
        self.get_attr = get_attr
        self.get_new2canon = get_new2canon

        can_copy = 'src_map' in fields and 'alignment' in fields

        def reader(dat):
            for d in dat[1]:
                yield {dat[0]: d}

        read_iters = [r.read(dat[1], dat[0], dir_) if r is not None else
                 reader(dat) for r, dat, dir_
                      in zip(readers, data, dirs)]


        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field)
                self.src_vocabs.append(src_ex_vocab)
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for k, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])
        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def __getitem__(self, idx):
        ex = copy.deepcopy(self.examples[idx])
        if not getattr(self, 'random_mask', False) and not getattr(self,
                'contrastive', False) and not getattr(self, 'get_adj', False) and \
                not getattr(self, 'get_attr', False) and not getattr(self,
                        'get_new2canon', False):
            #if isinstance(ex.tgt, (float, np.int64)):
            ex.src = [['<MASK>'] + ex.src[0]]
            return ex

        n_smiles_aug = getattr(self, 'n_smiles_aug', 1)
        src = copy.deepcopy(ex.src[0])
        try:
            tgt = copy.deepcopy(ex.tgt[0])
        except:
            pass
        src_smiles = ''.join(src)
        src_mol, src_adj, src_khop, src_atom_features = \
                            get_mol_and_adj(src_smiles, K=4)
        atom_features = atom_featurizer(src_mol)
        n_src_atom = len(src_adj)
        src_adj[np.arange(n_src_atom), np.arange(n_src_atom)] = -1
        src_khop[np.arange(n_src_atom), np.arange(n_src_atom)] = -1
        src_is_atomic = find_atomic_token(src)
        src_atomic_ind = src_is_atomic.nonzero()[0]
        adjs = []
        khops = []
        new2canons = []
        #if n_smiles_aug > 1:
        new_srcs = []
        featuress = {key: [] for key in src_atom_features}
        for _ in range(n_smiles_aug):
            new_smiles, new2canon = enum_smiles_from_mol(src_mol)
            new_srcs.append(smi_tokenizer(new_smiles).strip().split())
            n_token = len(new_srcs[-1])
            is_atomic = find_atomic_token(new_srcs[-1])
            atomic_ind = is_atomic.nonzero()[0]
            adj = torch.full((n_token, n_token), -1)
            khop = torch.full((n_token, n_token), -1)
            adj[atomic_ind.reshape(-1, 1), atomic_ind] = \
                torch.tensor(src_adj[new2canon.reshape(-1, 1),
                    new2canon]).float()
            khop[atomic_ind.reshape(-1, 1), atomic_ind] = \
                torch.tensor(src_khop[new2canon.reshape(-1, 1),
                    new2canon]).float()
            adjs.append(adj)
            khops.append(khop)
            for key, val in src_atom_features.items():
                featuress[key].append(torch.full((n_token,), -1))
                featuress[key][-1][is_atomic] = torch.tensor(val[new2canon]).float()
                featuress[key][-1] = featuress[key][-1].view(-1, 1, 1)
            tmp = torch.full((n_token,), -1)
            tmp[is_atomic] = torch.tensor(src_atomic_ind[new2canon]).float()
            new2canons.append(tmp.view(-1, 1, 1))
            featuress['khop'] = khops
            featuress['adj'] = adjs
        ex.src = new_srcs
        try:
            tgt_smiles = ''.join(tgt)
            tgt_mol, _, _, _ = \
                    get_mol_and_adj(tgt_smiles, K=1)
            ex.tgt = [smi_tokenizer(enum_smiles_from_mol(tgt_mol)[0]).strip().split()
                    for _ in range(n_smiles_aug)]
            #ex.tgt = [tgt for _ in range(n_smiles_aug)]
        except:
            pass
        for key, val in featuress.items():
            setattr(ex, key, val)
            #print(key, getattr(ex, key).unsqueeze())
        #input()
        #else:
        #    new_srcs = [src]
        #    n_token = len(new_srcs[-1])
        #    is_atomic = find_atomic_token(new_srcs[-1])
        #    atomic_ind = is_atomic.nonzero()[0]
        #    adj = torch.full((n_token, n_token), -1)
        #    adj[atomic_ind.reshape(-1, 1), atomic_ind] = src_adj
        #    adjs.append(adj)
        #    tmp = torch.full((n_token,), -1)
        #    tmp[is_atomic] = torch.tensor(src_atomic_ind)
        #    new2canons.append(tmp.view(-1, 1, 1))
        #ex.adj = adjs
        ex.new2canon = new2canons

        if not getattr(self, 'random_mask', False) and not getattr(self, 'contrastive', False):
            return ex

        if getattr(self, 'contrastive', False):
            new_srcs = [['<MASK>'] + src_ for src_ in new_srcs]
            ex.src = new_srcs
            return ex
        # masked lm
        src_field = self.fields['tgt']
        pad_token = src_field.base_field.pad_token

        srcs = []
        tgts = []
        for src in ex.src:
            tgt = src
            probs = np.random.rand(len(src))
            edit_ind = (probs < 0.15).nonzero()[0]
            remainder_ind = (probs >= 0.15).nonzero()[0]
            for ind in remainder_ind:
                tgt[ind] = pad_token
            #print("tgt", tgt)
            probs = probs[edit_ind] / 0.15
            for ind in (probs < 0.8).nonzero()[0]:
                src[edit_ind[ind]] = '<MASK>'
            vocab_list = list(src_field.fields[0][1].vocab.stoi.keys())
            vocab_list.remove('<s>')#src_field.base_field.init_token)
            vocab_list.remove('</s>')#src_field.base_field.eos_token)
            vocab_list.remove('<blank>')#src_field.base_field.pad_token)
            vocab_list.remove('<unk>')
            vocab_list.remove('<MASK>')
            for ind, val in zip((probs>0.9).nonzero()[0],
                    np.random.choice(vocab_list, len(probs[probs>0.9]))):
                src[edit_ind[ind]] = val
            srcs.append(src)
            tgts.append(tgt)
        ex.src = srcs
        ex.tgt = tgts
        return ex

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
