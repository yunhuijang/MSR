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

from util_cot import canonicalize, map_ring_cot, map_multiset_cot, map_token_name, map_carbon_chain_length
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
    
    # ring_cc, ring_type, ring_info = [], [], []
    # multi_cc, multi_type, multi_info = [], [], []
    # arom_info = []
    result = []
    
    cot_modes = cot_mode.split('-')
    for i, mode in enumerate(cot_modes):
        is_only_count = False
        print(f'Analysis for {mode}')
        cur_predicted_cot_list = [pred.split('.')[i]+'.' if len(pred.split('.'))>i else "" for pred in predicted_cot_list]
        cur_gt_cot_list = [gt.split('.')[i]+'.' for gt in gt_cot_list]
        if mode == 'ring':
            gt_info_list = [map_ring_size_from_cot(gt) for gt in cur_gt_cot_list]
            pred_info_list = [map_ring_size_from_cot(pred) for pred in cur_predicted_cot_list]
        elif ('simple' in mode) or ('full' in mode):
            gt_info_list = [map_multiset_from_cot(gt) for gt in cur_gt_cot_list]
            pred_info_list = [map_multiset_from_cot(pred) for pred in cur_predicted_cot_list]
        elif 'form' in mode:
            gt_info_list = [map_form_from_cot(gt) for gt in cur_gt_cot_list]
            pred_info_list = [map_form_from_cot(pred) for pred in cur_predicted_cot_list]
        elif mode == 'arom':
            gt_info_list = [map_arom_num_from_cot(gt) for gt in cur_gt_cot_list]
            pred_info_list = [map_arom_num_from_cot(pred) for pred in cur_predicted_cot_list]
            is_only_count = True
        elif mode == 'chain':
            gt_info_list = [map_chain_from_cot(gt) for gt in cur_gt_cot_list]
            pred_info_list = [map_chain_from_cot(pred) for pred in cur_predicted_cot_list]
            is_only_count = True
        
        acc_list = generate_correct_list(gt_info_list, pred_info_list, is_only_count)
        result.append(acc_list)
    return result
        
        
    
def compute_cot_alignment_smiles(predicted_cot_list, predicted_smiles_list, cot_mode='ring'):
    '''
    Compare the predicted CoT and predicted SMILES if it aligns or not
    '''
    
    ring_cot_list = map_ring_cot(predicted_smiles_list)
    ring_cot_list = [ring_cot[1:] for ring_cot in ring_cot_list]
    multiset_cot_list = map_multiset_cot(predicted_smiles_list)
    multiset_cot_list = [mul_cot[1:] for mul_cot in multiset_cot_list]
    
    ring_correct_list = []
    multiset_correct_list = []
    
    if len(cot_mode.split('-')) > 1:
        predicted_cot_list_ring = [pred.split('-')[0] for pred in predicted_cot_list]
        predicted_cot_list_multiset = [pred.split('-')[1] for pred in predicted_cot_list]
    else:
        predicted_cot_list_ring = predicted_cot_list
        predicted_cot_list_multiset = predicted_cot_list
    
    if 'ring' in cot_mode:
        ring_correct_list = [gt == pred for gt, pred in zip(ring_cot_list, predicted_cot_list_ring)]
        ring_accuracy = sum(ring_correct_list)/len(ring_cot_list)
        print(f"Ring alignment with CoT: {ring_accuracy}")
        
    elif ('simple' in cot_mode) or ('full' in cot_mode):
        multiset_correct_list = [gt == pred for gt, pred in zip(multiset_cot_list, predicted_cot_list_multiset)]
        multiset_accuracy = sum(multiset_correct_list)/len(multiset_cot_list)
        print(f"Multiset alignment with CoT: {multiset_accuracy}")

    return ring_correct_list, multiset_correct_list
    
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