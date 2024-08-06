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
import logging
import pubchempy
import json
from itertools import combinations
import collections
import itertools

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
    if mode not in ['simple', 'formula', 'full', 'only_type']:
        raise ValueError("Mode should be one of 'simple', 'formula', 'full'")
    
    
    if mode == 'formula':
        multiset_cot = [f" The molecular formula is {Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))}." for smiles in smiles_list]
        
    else:
        multiset_cot = []
        token_name_dict = map_token_name()
        
        
        tokens = [tokenize(smiles)[1:-1] for smiles in smiles_list]
        if mode == 'only_type':
            token_set = [set(token) for token in tokens]
            token_set = [[t for t in token if t in set(NODE_TOKENS).union(BOND_TOKENS)] for token in token_set]
            return [" It includes " + ", ".join(token) + '.' if len(token) > 0 else "" for token in token_set]
        else:
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

def map_symbol(mol, index, i, ring_size):
    atom = mol.GetAtoms()[index]
    is_aromatic = atom.GetIsAromatic()
    if is_aromatic:
        symbol =  atom.GetSymbol().lower()
    else:
        symbol = atom.GetSymbol()
    if i == 0 or i == ring_size-1:
        symbol = symbol + '1'
    return symbol

def smiles_to_iupac(smiles):
    try:
        compounds = pubchempy.get_compounds(smiles, namespace='smiles')
    except:
        logging.warning(f"Error in mapping SMILES: {smiles}")
        return ""
    m = compounds[0]
    if m.iupac_name is None:
        return ""
    else:
        return m.iupac_name

def map_ring_name_cot(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    ring_info = [mol.GetRingInfo().AtomRings() if mol is not None else [] for mol in mols]
    ring_smiles = [[Chem.rdmolfiles.MolFragmentToSmiles(mol, atomsToUse=s) for s in ri] for mol, ri in zip(mols, ring_info)]
    with open('resource/data/ring_to_iupac.json', 'r') as fp:
        ring_name_dict = json.load(fp)
    ring_info_multiset_iupac = [[ring_name_dict.get(smi, "") for smi in smis] for smis in ring_smiles]
    ring_info_count = [Counter(ri) for ri in ring_info_multiset_iupac]
    ring_cot = []
    for srs in ring_info_count:
        cot = " It includes"
        for key, value in srs.items():
            if value == 1:
                cot += f" {value} {key} ring,"
            else:
                cot += f" {value} {key} rings,"
        cot = cot[:-1] + '.'
        if cot == " It include.":
            cot = " It does not include any rings."

        cot.replace("  ", " ")
        ring_cot.append(cot)
        
    return ring_cot

def map_iupac_cot(smiles_list):
    iupac_list = [smiles_to_iupac(smi) for smi in tqdm(smiles_list)]
    
    cot_list = [f" The IUPAC form is {iupac}." if len(iupac)>0 else " The IUPAC form is not available." for iupac in iupac_list]
    
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

def merge_connected_ring(ring_set):
    result, visited = set(), set()
    components = collections.defaultdict(list)
    adj_list = collections.defaultdict(list)

    def dft(node, key):
        visited.add(node)
        components[key].append(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                dft(neighbor, key)

    for r1, r2 in itertools.combinations_with_replacement(ring_set, 2):
        r1 = frozenset(r1)
        r2 = frozenset(r2)
        if r1 & r2:
            adj_list[r1].append(r2)
            adj_list[r2].append(r1)
    for node in adj_list:
        if node not in visited:
            dft(node, node)

    for node, neighbors in components.items():
        result.add(node.union(*neighbors))
        
    return result

def get_subs(mol, final_ring_set):
    if len(final_ring_set) > 0:
        atom_indices = [tuple(set().union(*s)) for s in final_ring_set]
        substructures = [Chem.rdmolfiles.MolFragmentToSmiles(mol, atomsToUse=atom_index) for atom_index in atom_indices]
        return substructures
    else:
        return []

def get_ring_substructure(ri, mol):
    '''
    To derive spiro, fused, and bridge ring substructures from molecule
    '''
    spiro_set = set()
    fused_set = set()
    bridge_set = set()
    for ring_1, ring_2 in combinations(ri, 2):
        sharing_atom = set(ring_1) & set(ring_2)
        if len(sharing_atom) == 1:
            spiro_set.add(tuple(sorted([ring_1, ring_2])))
        elif len(sharing_atom) == 2:
            fused_set.add(tuple(sorted([ring_1, ring_2])))
        elif len(sharing_atom) > 3:
            bridge_set.add(tuple(sorted([ring_1, ring_2])))
    
    final_fused_set = merge_connected_ring(fused_set)
    final_spiro_set = merge_connected_ring(spiro_set)
    final_bridge_set = merge_connected_ring(bridge_set)
    substructures_fused = get_subs(mol, final_fused_set)
    substructures_spiro = get_subs(mol, final_spiro_set)
    substructures_bridge = get_subs(mol, final_bridge_set)
    
    return substructures_spiro, substructures_fused, substructures_bridge
        
def get_connected_ring_name(ri, mol):
    
    if len(ri) == 0:
        return []
    substructures_spiro, substructures_fused, substructures_bridge = get_ring_substructure(ri, mol)
    result_substructures = substructures_fused + substructures_spiro + substructures_bridge
    final_result = []
    
    with open('resource/data/connected_ring_to_iupac.json', 'r') as fp:
        ring_name_dict = json.load(fp)
    
    for smi in result_substructures:
        iupac = ring_name_dict.get(smi, "unknown")
        if iupac is None:
            sub_mol = Chem.MolFromSmiles(smi)
            sub_ring_info = sub_mol.GetRingInfo().AtomRings()
            sub_rings = [Chem.rdmolfiles.MolFragmentToSmiles(sub_mol, atomsToUse=s) for s in sub_ring_info]
            final_result.extend([ring_name_dict.get(sub_ring, "") for sub_ring in sub_rings])
        else:
            final_result.append(iupac)
        
    # print(final_result)
    return final_result

def map_connected_ring_name_cot(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    ring_info = [mol.GetRingInfo().AtomRings() for mol in mols]
    ring_connectivity = [get_connected_ring_name(ri, mol) for ri, mol in zip(tqdm(ring_info), mols)]
    ring_info_count = [Counter(ri) for ri in ring_connectivity]
    ring_cot = []
    for srs in ring_info_count:
        cot = " It includes"
        for key, value in srs.items():
            if value == 1:
                cot += f" {value} {key} ring,"
            else:
                cot += f" {value} {key} rings,"
        cot = cot[:-1] + '.'
        if cot == " It include.":
            cot = " It does not include any rings."

        cot = cot.replace("  ", " ")
        ring_cot.append(cot)
        
    return ring_cot

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
    # <FIX> Need to be fixed when CoT added
    cot_mode_multiset = hparams.cot_mode_multiset
    cot_mode_ring = hparams.cot_mode_ring
    cot_mode_fragment = hparams.cot_mode_fragment
    cot_mode_aromaticity = hparams.cot_mode_aromatic
    cot_mode_chain = hparams.cot_mode_chain
    cot_mode_ring_name = hparams.cot_mode_ring_name
    cot_mode_iupac = hparams.cot_mode_iupac
    cot_mode_con_ring_name = hparams.cot_mode_con_ring_name
    # CoT order: chain, fragment, ring, multiset, aromatic, ring_name, connected_ring_name, iupac
    cot_mode = ""
    if cot_mode_chain:
        cot_mode += '-chain'
    if cot_mode_fragment:
        cot_mode += '-frag'
    if cot_mode_ring:
        cot_mode += '-ring'
    if cot_mode_multiset in ['simple', 'full', 'formula', 'only_type']:
        cot_mode += f'-multiset_{cot_mode_multiset}'
    if cot_mode_aromaticity:
        cot_mode += '-arom'
    if cot_mode_ring_name:
        cot_mode += '-rname'
    if cot_mode_con_ring_name:
        cot_mode += '-conrna'
    if cot_mode_iupac:
        cot_mode += '-iupac'
        
    return cot_mode

def add_cot_to_target(examples, targets, cot_mode):
    # CoT order: chain, fragment, ring, multiset, aromatic, ring_name, iupac
    # <FIX> Need to be fixed when CoT added
    if 'iupac' in cot_mode:
        targets = [f"{cot_iupac}{target}" for target, cot_iupac in zip(targets, examples['cot_iupac'])]
    
    if 'conrna' in cot_mode:
        targets = [f"{cot_con_ring_name}{target}" for target, cot_con_ring_name in zip(targets, examples['cot_connected_ring_name'])]
    
    if 'rname' in cot_mode:
        targets = [f"{cot_ring_name}{target}" for target, cot_ring_name in zip(targets, examples['cot_ring_name'])]
    
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