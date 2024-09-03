import pandas as pd
from itertools import compress
import numpy as np
from pathlib import Path


from analysis import compute_cot_accuracy, compute_cot_alignment_smiles, get_confusion_matrix, map_multiset_from_cot
from util_cot import map_ring_cot, map_multiset_cot
import os

cot_mode = 'chain'


smiles_list_path = os.path.join('predictions/two_stage_ft_cot/reasoning', f'molt5-base-{cot_mode}.txt')
smiles_pair_list = [
[pair.split('\t')[:-2][0], pair.split('\t')[-2], pair.split('\t')[-1]] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()[1:]
]
# if self.hparams.test:
#     smiles_pair_list = smiles_pair_list[:20]
description_list = [pair[0] for pair in smiles_pair_list]
gt_cot = [pair[1] for pair in smiles_pair_list]
predicted_cot = [pair[2] for pair in smiles_pair_list]

# cot_mode = map_cot_mode()
if cot_mode[0] == '-':
    cot_mode = cot_mode[1:]
# ring_acc, multi_acc, arom_acc = compute_cot_accuracy(gt_cot, predicted_cot, cot_mode=cot_mode)
cot_acc = compute_cot_accuracy(gt_cot, predicted_cot, cot_mode=cot_mode)
wandb_log_dict = {}
cot_modes = cot_mode.split('-')
for mode, acc in zip(cot_modes, cot_acc):
    if type(acc) == list:
        wandb_log_dict[f'cot/{mode}_acc'] = sum(acc)/len(acc)
    else:
        # tuple (tuple of 3 lists)
        wandb_log_dict[f'cot/{mode}_acc_count'] = sum(acc[0])/len(acc[0])
        wandb_log_dict[f'cot/{mode}_acc_type'] = sum(acc[1])/len(acc[0])
        wandb_log_dict[f'cot/{mode}_acc'] = sum(acc[2])/len(acc[0])

print(wandb_log_dict)