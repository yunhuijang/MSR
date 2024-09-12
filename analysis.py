import os
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import argparse
import wandb
from os.path import join
import pandas as pd
from rdkit import Chem
from collections import Counter
import numpy as np
import re
from itertools import compress
import logging

from util_cot import *
from tokens import tokenize, NODE_TOKENS, BOND_TOKENS
import torch
from torch.nn.utils.rnn import pad_sequence




def compare_smiles(architecture, task):
    smiles_list_path = os.path.join('predictions/cot', f"{architecture}-{task}.txt")
    smiles_pair_list = [
    [" ".join(pair.split()[:-2]), pair.split()[-2], pair.split()[-1]] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
    ][1:]
    description_list = [pair[0] for pair in smiles_pair_list]
    gt_smiles_list = [pair[1] for pair in smiles_pair_list]
    prediction_list = [pair[2] for pair in smiles_pair_list]


    df_wrong = pd.DataFrame([gt_smiles_list, prediction_list]).T
    df_wrong.columns = ['SRC', 'TGT']

    df_wrong['SRC_MOL'] = df_wrong['SRC'].apply(lambda x: Chem.MolFromSmiles(x))
    df_wrong['TGT_MOL'] = df_wrong['TGT'].apply(lambda x: Chem.MolFromSmiles(x))
    df_wrong['SRC_RING'] = df_wrong['SRC_MOL'].apply(lambda x: x.GetRingInfo().AtomRings() if x is not None else [])
    df_wrong['TGT_RING'] = df_wrong['TGT_MOL'].apply(lambda x: x.GetRingInfo().AtomRings() if x is not None else [])
    df_wrong['SRC_RING_COUNT'] = df_wrong['SRC_RING'].apply(lambda x: len(x))
    df_wrong['TGT_RING_COUNT'] = df_wrong['TGT_RING'].apply(lambda x: len(x))
    df_wrong['SRC_RING_SIZE'] = df_wrong['SRC_RING'].apply(lambda x: Counter([len(s) for s in x]))
    df_wrong['TGT_RING_SIZE'] = df_wrong['TGT_RING'].apply(lambda x: Counter([len(s) for s in x]))
    df_wrong['RING_SIZE'] = df_wrong['TGT_RING_SIZE'] == df_wrong['SRC_RING_SIZE']
    df_wrong['RING_COUNT'] = df_wrong['TGT_RING_COUNT'] == df_wrong['SRC_RING_COUNT']

    src = df_wrong['SRC'].apply(lambda x: Counter(tokenize(x)))
    src_list = [{key: value for key, value in tc.items() if key in set(NODE_TOKENS).union(BOND_TOKENS)} for tc in src]
    # src_token_count = pad_sequence([s[3:3+len(NODE_TOKENS)+len(BOND_TOKENS)] for s in src]).T
    tgt = df_wrong['TGT'].apply(lambda x:Counter(tokenize(x)))
    tgt_list = [{key: value for key, value in tc.items() if key in set(NODE_TOKENS).union(BOND_TOKENS)} for tc in tgt]
    # tgt_token_count = pad_sequence([s[3:3+len(NODE_TOKENS)+len(BOND_TOKENS)] for s in tgt]).T

    correct_multiset = Counter([s == t for s, t in zip(src_list, tgt_list)])[True]
    correct_ring_count = df_wrong['RING_COUNT'].sum()
    correct_ring_size = df_wrong['RING_SIZE'].sum()

    print(f"Correct Multiset: {correct_multiset/len(df_wrong)}")
    print(f"Correct Ring Count: {correct_ring_count/len(df_wrong)}")
    print(f"Correct Ring Size: {correct_ring_size/len(df_wrong)}")
    

def map_ring_count_from_cot(cot):
    if cot == 'It does not include any rings.':
        return 0
    else:
        return cot.count(',')+1

def map_ring_size_from_cot(cot):
    
    cot_splitted = cot.split(' ')
    
    of_index = [index for index, word in enumerate(cot_splitted) if word == 'of']
    ring_number_index = [oi-2 for oi in of_index]
    ring_size_index = [oi+2 for oi in of_index]
    ring_dict = {}
    for rsi, rni in zip(ring_size_index, ring_number_index):
        if (len(cot_splitted) > rsi) and (len(cot_splitted) > rni):
            if cot_splitted[rsi][:-1].isdigit() and cot_splitted[rni].isdigit():
                ring_size = int(cot_splitted[rsi][:-1])
                ring_number = int(cot_splitted[rni])
            
                ring_dict[ring_size] = ring_number
    
    return dict(sorted(ring_dict.items()))

def map_arom_num_from_cot(cot):
    if cot == ' It does not include any aromatic ring.':
        return 0
    else:
        try:
            if cot[0] == ' ':
                return int(cot.split(' ')[3])
            else:
                return int(cot.split(' ')[2])
        except:
            logging.warning(f"Error in mapping aromatic ring number from CoT: {cot}")
            return 100
    
def map_chain_from_cot(cot):
    try:
        return int(cot.split(' ')[-1][:-1])
    except:
        logging.warning(f"Error in mapping chain length from CoT: {cot}")
        return 100

def map_multiset_from_cot(cot):
    
    cot_splitted = cot.split(' ')
    count_indices = [index for index, word in enumerate(cot_splitted) if word.isdigit()]
    count_list = []
    type_list = []
    type_count_dict = {}
    for start, end in zip(count_indices, count_indices[1:] + [-1]):
        count_list.append(int(cot_splitted[start]))
        if end == -1:
            t = " ".join(cot_splitted[start+1:])[:-1]
            # type_list.append()
        else:
            t = " ".join(cot_splitted[start+1:end])[:-1]
            # type_list.append()
        if len(t) > 0:
            if t[-1] == 's':
                t = t[:-1]
            type_list.append(t)
        
    for type, count in zip(type_list, count_list):
        type_count_dict[type] = count
    
    return dict(sorted(type_count_dict.items()))

def map_form_from_cot(cot):
    form = cot.split(' ')[-1][:-1]
    multiset = [s for s in re.findall(r'[a-zA-Z]+', form)]
    count = [int(s) for s in re.findall(r'[\d]+', form)]
    type_count_dict = {}
    for type, count in zip(multiset, count):
        type_count_dict[type] = count
    return dict(sorted(type_count_dict.items()))

def map_type_from_cot(cot):
    try:
        types = cot[len(" It includes "):-1]
        type_set = [t for t in types.split(',') if len(t)>0]
        return set([t if t[0]!= ' ' else t[1:] for t in type_set])
    except:
        return set()


def map_iupac_from_cot(cot):
    return cot[len(" The IUPAC form is "):-1]

def map_ring_size_from_cot(cot):
    
    cot_splitted = cot.split(' ')
    
    of_index = [index for index, word in enumerate(cot_splitted) if word == 'of']
    ring_number_index = [oi-2 for oi in of_index]
    ring_size_index = [oi+2 for oi in of_index]
    ring_dict = {}
    for rsi, rni in zip(ring_size_index, ring_number_index):
        if (len(cot_splitted) > rsi) and (len(cot_splitted) > rni):
            if cot_splitted[rsi][:-1].isdigit() and cot_splitted[rni].isdigit():
                ring_size = int(cot_splitted[rsi][:-1])
                ring_number = int(cot_splitted[rni])
            
                ring_dict[ring_size] = ring_number
    
    return dict(sorted(ring_dict.items()))

def map_ring_name_from_cot(cot):
    cot_splitted = cot.split(' ')
    ring_index = [index for index, word in enumerate(cot_splitted) if 'ring' in word]
    ring_number_index = [oi-2 for oi in ring_index]
    ring_name_index = [oi-1 for oi in ring_index]
    ring_dict = {}
    for rnum, rname in zip(ring_number_index, ring_name_index):
        if (len(cot_splitted) > rnum) and (len(cot_splitted) > rname):
            if cot_splitted[rnum].isdigit():
                ring_number = int(cot_splitted[rnum])
                ring_name = cot_splitted[rname]
                
                ring_dict[ring_name] = ring_number
    return dict(sorted(ring_dict.items()))
    
def map_scaffold_from_cot(cot):
    if "The scaffold is" in cot:
        scaffold = cot[len(" The scaffold is "):-1]
    else:
        scaffold = ""
    return scaffold

def map_functional_group_from_cot(cot):
    if "The functional group of the molecule is" in cot:
        functional_groups = cot[len(" The functional group of the molecule is "):-1]
        fgs = set(functional_groups.split(','))
        fgs = [fg.strip() for fg in fgs]
    else:
        fgs = set()
    return sorted(fgs)

def generate_correct_list(gt_info_list, pred_info_list, is_only_count=False):
    # whole information of rings
    info_correct_list = [gt == pred for gt, pred in zip(gt_info_list, pred_info_list)]
    print(f"Accuracy: {sum(info_correct_list)/len(gt_info_list)}")
    if is_only_count:
        return info_correct_list
    # total number of rings
    count_correct_list = [len(gt) == len(pred) for gt, pred in zip(gt_info_list, pred_info_list)]
    # type of ring sizes
    type_correct_list = [set(gt.keys()) == set(pred.keys()) for gt, pred in zip(gt_info_list, pred_info_list)]

    
    print(f"Type Accuracy: {sum(type_correct_list)/len(gt_info_list)}")
    print(f"Count Accuracy: {sum(count_correct_list)/len(gt_info_list)}")
    
    return count_correct_list, type_correct_list, info_correct_list


def compute_cot_accuracy(gt_cot_list, predicted_cot_list, cot_mode='ring'):
    '''
    Compare the ground-truth CoT to predicted CoT
    '''
    # <FIX> Need to be fixed when CoT added
    result = []
    
    cot_modes = cot_mode.split('-')
    for i, mode in enumerate(cot_modes):
        is_only_count = False
        print(f'Analysis for {mode}')
        cur_predicted_cot_list = [pred.split('.')[i]+'.' if len(pred.split('.'))>i else "" for pred in predicted_cot_list]
        cur_gt_cot_list = [gt.split('.')[i]+'.' for gt in gt_cot_list]
        
        cot_function_dict = {'func_simple': map_functional_group_from_cot, 'func_smiles': map_functional_group_from_cot, 'scaffold': map_scaffold_from_cot, \
                        'chain': map_chain_from_cot, 'fragment': map_fragment_cot, 'ring': map_ring_size_from_cot, 'multiset_simple': map_multiset_from_cot, \
                        'multiset_full': map_multiset_from_cot, 'multiset_formula': map_form_from_cot, 'multiset_type': map_type_from_cot, \
                        'aromatic': map_arom_num_from_cot, 'ring_name': map_ring_name_from_cot, 'con_ring_name': map_ring_name_from_cot, \
                        'iupac': map_iupac_from_cot, 'double_bond': map_num_double_bond}
        
        gt_info_list = [cot_function_dict.get(mode)(gt) for gt in cur_gt_cot_list]
        pred_info_list = [cot_function_dict.get(mode)(gt) for gt in cur_predicted_cot_list]
        if mode in ['multiset_type', 'aromatic', 'chain', 'iupac', 'scaffold', 'func_simple', 'func_smiles']:
            is_only_count = True
        
        
        acc_list = generate_correct_list(gt_info_list, pred_info_list, is_only_count)
        result.append(acc_list)
    return result
        
        
    
def compute_cot_alignment_smiles(predicted_cot_list, predicted_smiles_list, cot_mode='ring'):
    '''
    Compare the predicted CoT and predicted SMILES if it aligns or not
    '''
    
    function_map_dict = {'ring': map_ring_cot, 'simple': map_multiset_cot, 'full': map_multiset_cot,
                        'form': map_multiset_cot, 'only_type': map_multiset_cot, 'arom': map_aromatic_ring_cot,
                        'chain': map_carbon_chain_length, 'iupac': map_iupac_cot, 'rname': map_ring_name_cot,
                        'conrna': map_connected_ring_name_cot, 'scaffold': map_scaffold_cot, 'fg': map_functional_group_cot}
    func = function_map_dict[cot_mode]
    gt_cot_list = func(predicted_smiles_list)
    gt_cot_list = [gt_cot[1:] for gt_cot in gt_cot_list]
    # gt_cot_list = [gt_cot[1:] for gt_cot in gt_cot_list]
    correct_list = [gt == pred for gt, pred in zip(gt_cot_list, predicted_cot_list)]
    print(f"Accuracy: {sum(correct_list)/len(gt_cot_list)}")


    return correct_list
    
def get_confusion_matrix(cot_correct_list, align_correct_list):
    '''
    Get the confusion matrix of the CoT and SMILES alignment
    '''
    correct_both = sum([correct_cot and correct_align for correct_cot, correct_align in zip(cot_correct_list, align_correct_list)])
    wrong_cot = sum([correct_cot and correct_align for correct_cot, correct_align in zip(~np.array(cot_correct_list), np.array(align_correct_list))])
    wrong_smiles_align = sum([correct_cot and correct_align for correct_cot, correct_align in zip(np.array(cot_correct_list), ~np.array(align_correct_list))])
    wrong_both = sum([correct_cot and correct_align for correct_cot, correct_align in zip(~np.array(cot_correct_list), ~np.array(align_correct_list))])

    print([[correct_both, wrong_cot], [wrong_smiles_align, wrong_both]])
    print('Normalized')
    print([[correct_both/len(cot_correct_list), wrong_cot/len(cot_correct_list)], 
          [wrong_smiles_align/len(cot_correct_list), wrong_both/len(cot_correct_list)]])