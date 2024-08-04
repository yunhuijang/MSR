import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from collections import Counter
from tqdm import tqdm
import re
from torch.nn.utils.rnn import pad_sequence
import os
from pathlib import Path
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
from rdkit.Chem import MCS


from tokens import NODE_TOKENS, BOND_TOKENS, tokenize, id_to_token



def map_token_name():
    '''
    To map CoT tokens to their natural language names
    '''
    # TODO: should this be one-to-one mapping? (e.g., /, \\ are duplicated)
    token_name_dict = {}
    node_name_dict = {"C": "carbon", "F": 'fluorine', "O": "oxygen", "N": "nitrogen", "H": "hydrogen",
                      "Br": 'bromine', "Cl": 'chlorine', "S": 'sulfur', "P": 'phosphorus', "I": "iodine",
                      "c": "aromatic carbon", "n": "aromatic nitrogen", "o": "aromatic oxygen", "s": "aromatic sulfur"}
    bond_name_dict = {"-": "single", "=": "double", "#": "triple", "/": "single bond adjacent to double", "\\": "single bond adjacent to double"}
    
    for node_token in NODE_TOKENS:
        name = ""
        hydrogen_flag = False
        if node_token[0] == '[':
            if '@' in node_token:
                name += "chiral "
            name += node_name_dict[node_token[1:2]]
            
            if 'H2' in node_token:
                name += ' with two hydrogen atoms'
                hydrogen_flag = True
            elif 'H' in node_token:
                name += ' with one hydrogen atom'
                hydrogen_flag = True
                
            if hydrogen_flag:
                name += ' and'
            else:
                name += ' with'
            
            if node_token[-2] == '+':
                name += " a positive charge"
            elif node_token[-2] == '-':
                name += " a negative charge"
            else:
                name += " no charge"
                
            token_name_dict[node_token] = name
        else:
            token_name_dict[node_token] = node_name_dict[node_token]
    
    for edge_token in BOND_TOKENS:
        token_name_dict[edge_token] = bond_name_dict[edge_token]
    
    for key, value in token_name_dict.items():
        if key in node_name_dict.keys():
            token_name_dict[key] = f"{value} atom"
        elif key in bond_name_dict.keys():
            token_name_dict[key] = f"{value} bond"
    
    return token_name_dict


def map_multiset_cot(smiles_list, mode='simple'):
    '''
    SMILES -> Multiset CoT
    '''
    if mode not in ['simple', 'formula', 'full']:
        raise ValueError("Mode should be one of 'simple', 'formula', 'full'")
    
    
    if mode == 'formula':
        multiset_cot = [f" The molecular formula is {Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))}." for smiles in smiles_list]
    else:
        multiset_cot = []
        token_name_dict = map_token_name()
        
        
        tokens = [tokenize(smiles)[1:-1] for smiles in smiles_list]
        token_count = [Counter(token) for token in tokens]
        multiset_list = [{key: value for key, value in tc.items() if key in set(NODE_TOKENS).union(BOND_TOKENS)} for tc in token_count]
        
        for multiset in multiset_list:
            cot = " It includes"
            for key, value in multiset.items():
                if mode == 'simple':
                    if value == 1:
                        cot += f" {value} {key},"
                    else:
                        cot += f" {value} {key}s,"
                else:
                    if value == 1:
                        cot += f" {value} {token_name_dict[key]},"
                    else:
                        cot += f" {value} {token_name_dict[key]}s,"
                    
            cot = cot[:-1] + '.'
            multiset_cot.append(cot)

    return multiset_cot

def map_ring_cot(smiles_list):
    '''
    SMILES -> Ring Count CoT
    '''
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    ring_info = [mol.GetRingInfo().AtomRings() if mol is not None else [] for mol in mols]
    ring_size = [Counter([len(s) for s in ri]) for ri in ring_info]
    ring_cot = []
    for srs in ring_size:
        cot = " It includes"
        for key, value in srs.items():
            if value == 1:
                cot += f" {value} ring of size {key},"
            else:
                cot += f" {value} rings of size {key},"
        cot = cot[:-1] + '.'
        if cot == " It include.":
            cot = " It does not include any rings."

        
        ring_cot.append(cot)
    return ring_cot

def map_aromatic_ring_cot(smiles_list):
    arom_cot = []
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    aromatic_ring_num = [CalcNumAromaticRings(mol) if mol is not None else [] for mol in mols]
    for arom_num in aromatic_ring_num:
        cot = " It includes"
        if arom_num == 0:
            cot = " It does not include any aromatic ring."
        elif arom_num == 1:
            cot += f" {arom_num} aromatic ring."
        else:
            cot += f" {arom_num} aromatic rings."

        arom_cot.append(cot)
    
    return arom_cot

def map_carbon_chain_length(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    carbon_mol = Chem.MolFromSmiles('C'*100)
    carbon_chain_length = [MCS.FindMCS([mol, carbon_mol]).smarts for mol in mols]
    carbon_chain_length = [smart.count('[#6]') if smart is not None else 0 for smart in carbon_chain_length]
    cot_list = [f" The longest carbon chain length is {ccl}." for ccl in carbon_chain_length]
    
    return cot_list
    

def map_fragment_cot(split):
    '''
    SMILES -> Fragment CoT based on BRICS fragmentation
    '''
    # mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    # TODO: need to fix molecules that are not decomposed?
    # frag_list = [BRICS.BRICSDecompose(mol) if len(BRICS.BRICSDecompose(mol))>1 else set("") for mol in tqdm(mols, "Fragmentation")]
    
    # frag_list = [[re.sub(r'\d+', '', frag) for frag in fl] for fl in frag_list]
    smiles_path = os.path.join('ChEBI-20_data', f'{split}_fragment.txt')
    frag_list = [frag.split() for frag in Path(smiles_path).read_text(encoding="utf-8").splitlines()]

    frag_cot = []
    for fl in frag_list:
        cot = " It includes"
        for frag in fl:
            cot += f" {frag},"
        cot = cot[:-1] + '.'
        if cot == " It include.":
            cot = " It consists of a single fragment."
        frag_cot.append(cot)
    return frag_cot


def canonicalize(smiles, is_kekulize=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol, kekuleSmiles=is_kekulize)
    except:
        return None   


    if len(smiles) == 0:
        return None

    return smiles

def map_cot_mode(hparams):
    '''
    Map CoT mode to string
    '''
    
    cot_mode_multiset = hparams.cot_mode_multiset
    cot_mode_ring = hparams.cot_mode_ring
    cot_mode_fragment = hparams.cot_mode_fragment
    cot_mode_aromaticity = hparams.cot_mode_aromatic
    cot_mode_chain = hparams.cot_mode_chain
    # CoT order: chain, fragment, ring, multiset, aromatic
    cot_mode = ""
    if cot_mode_chain:
        cot_mode += '-chain'
    if cot_mode_fragment:
        cot_mode += '-frag'
    if cot_mode_ring:
        cot_mode += '-ring'
    if cot_mode_multiset in ['simple', 'full', 'formula']:
        cot_mode += f'-multiset_{cot_mode_multiset}'
    if cot_mode_aromaticity:
        cot_mode += '-arom'

        
    return cot_mode

def add_cot_to_target(examples, targets, cot_mode):
    # CoT order: chain, fragment, ring, multiset, aromatic
    if 'arom' in cot_mode:
        targets = [f"{cot_arom}{target}" for target, cot_arom in zip(targets, examples['cot_aromatic'])]
    
    if 'multiset' in cot_mode:
        targets = [f"{cot_multiset}{target}" for target, cot_multiset in zip(targets, examples['cot_multiset'])]
        
    if 'ring' in cot_mode:
        targets = [f"{cot_ring}{target}" for target, cot_ring in zip(targets, examples['cot_ring'])]
    
    if 'frag' in cot_mode:
        targets = [f"{cot_fragment}{target}" for target, cot_fragment in zip(targets, examples['cot_fragment'])]
    
    if 'chain' in cot_mode:
        targets = [f"{cot_fragment}{target}" for target, cot_fragment in zip(targets, examples['cot_chain'])]

    
    return targets