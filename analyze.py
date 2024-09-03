import pandas as pd
from itertools import compress
import numpy as np

from analysis import compute_cot_accuracy, compute_cot_alignment_smiles, get_confusion_matrix, map_multiset_from_cot
from util_cot import map_ring_cot, map_multiset_cot


cot_mode = 'conrna'
arch  ='molt5_base'
# df = pd.read_csv(f'resource/table/{arch}_cring.csv')
df_1 = pd.read_csv(f'resource/table/{arch}_cring_rea.csv')
df_2 = pd.read_csv(f'resource/table/{arch}_cring_ans.csv')

df = pd.concat([df_1, df_2], axis=1)

gt_cot_list = df['gt_cot'].tolist()
predicted_cot_list = df['predicted_cot'].tolist()
gt_smiles_list = df['gt_smiles'].tolist()
predicted_smiles_list = df['predicted_smiles'].tolist()

cot_correct_list = compute_cot_accuracy(gt_cot_list, predicted_cot_list, cot_mode=cot_mode)
cot_correct_list = cot_correct_list[0][0]

# keep only correct cot predictions
correct_predicted_cot_list = list(compress(predicted_cot_list, cot_correct_list))
correct_predicted_smiles_list = list(compress(predicted_smiles_list, cot_correct_list))



align_correct_list = compute_cot_alignment_smiles(predicted_cot_list, predicted_smiles_list, cot_mode=cot_mode)

get_confusion_matrix(cot_correct_list, align_correct_list)
