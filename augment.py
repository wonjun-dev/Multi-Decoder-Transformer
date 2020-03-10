import rdkit
import rdkit.Chem.rdmolfiles as R
import rdkit.Chem.Draw as D
import rdkit.Chem as C

import pandas as pd
import torch
import re


def augment_smiles(batch, vocab):
    """
    Args:
        batch: current batch which has attiributes about raw src (batch.data.examples[idx].src[0])
        vocab: src vocab used for constructing numericalized input
    Return:
        2 different augmented src smiles 
        batch_1: augmented smiles 1
        batch_2: augmented smiles 2
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    idxs = batch.indices
    max_len = 0
    src_vocab = vocab
    augs = []
    for i in idxs:
        raw_src = batch.dataset.examples[i].src[
            0]  # token seperated list w/o padding
        smi = ''.join(raw_src)  # smiles string
        mol = R.MolFromSmiles(smi)  # convert smi to mol
        smi_aug = C.MolToSmiles(mol,
                                doRandom=True)  # convert mol to augmented smi
        tokens = [token for token in regex.findall(smi_aug)
                  ]  # token seperated list w/o padding
        if len(tokens) > max_len:
            max_len = len(tokens)

        assert smi_aug == ''.join(tokens)

        augs.append(tokens)
        # aug_src = ' '.join(tokens)
    new_batch = _smi2tensor(augs, vocab, max_len)

    return new_batch


def _smi2tensor(augs, vocab, max_len):
    for cnt, tokens in enumerate(augs):
        stoi = [vocab[t] for t in tokens]
        if len(stoi) < max_len:
            num = max_len - len(stoi)
            for _ in range(num):
                stoi.append(1)  # zero padding {<blank> : 1}

        stoi = torch.tensor(stoi).unsqueeze(1)
        if cnt == 0:
            new_batch = stoi
        else:
            new_batch = torch.cat((new_batch, stoi), dim=1)

    return new_batch.unsqueeze(
        -1).cuda()  # (max_len, bs, 1)  same shape with original batch
