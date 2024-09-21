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
from rdkit.Chem import MCS, rdFMCS
import logging
import pubchempy
import json
from itertools import combinations
import collections
import itertools
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles 
from thermo import functional_groups
from inspect import getmembers, isfunction
from rdkit.Chem.rdchem import EditableMol
from rdkit.Chem import FindMolChiralCenters
from rdkit.Chem import rdMolDescriptors
import requests

from tokens import NODE_TOKENS, BOND_TOKENS, tokenize, id_to_token

TOTAL_COT_MODES = ['func_simple', 'func_smiles', 'scaffold', 'chain', 'fragment', 'ring', 'multiset_simple', \
            'multiset_full', 'multiset_formula', 'multiset_type', 'aromatic', 'ring_name',  \
            'con_ring_name', 'iupac', 'double_bond', 'chiral', 'weight', 'name', 'func_chem']


def flatten(xss):
    return [x for xs in xss for x in xs]

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
    if mode not in ['simple', 'formula', 'full', 'type']:
        raise ValueError("Mode should be one of 'simple', 'formula', 'full', and 'type'")
    
    
    if mode == 'formula':
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        multiset_cot = [f" The molecular formula is {Chem.rdMolDescriptors.CalcMolFormula(mol)}." if mol is not None else 'The molecular formula is unknown.' for mol in mol_list]
        
    else:
        multiset_cot = []
        token_name_dict = map_token_name()
        
        
        tokens = [tokenize(smiles)[1:-1] for smiles in smiles_list]
        if mode == 'type':
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
    aromatic_ring_num = [CalcNumAromaticRings(mol) if mol is not None else 0 for mol in mols]
    for arom_num in aromatic_ring_num:
        cot = " The molecule contains"
        if arom_num == 0:
            cot = " It does not have any aromatic ring."
        elif arom_num == 1:
            cot += f" {arom_num} aromatic ring."
        else:
            cot += f" {arom_num} aromatic rings."

        arom_cot.append(cot)
    
    return arom_cot

def map_carbon_chain_length(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    edit_mols = [EditableMol(mol) if mol is not None else None for mol in mols]
    carbon_mol = Chem.MolFromSmiles('C'*100)
    ring_atoms = [mol.GetRingInfo().AtomRings() if mol is not None else [] for mol in mols]
    ring_atom_indices = [sorted(list(set([atom for ring in ring_atom for atom in ring])), reverse=True) for ring_atom in ring_atoms]
    for edit_mol, atom_index in zip(edit_mols, ring_atom_indices):
        if edit_mol is None:
            continue
        for ai in atom_index:
            edit_mol.RemoveAtom(ai)
    mol_wo_rings = [edit_mol.GetMol() if edit_mol is not None else None for edit_mol in edit_mols]
    carbon_chain_length = [rdFMCS.FindMCS([mol, carbon_mol]).smartsString if mol is not None else "" for mol in mol_wo_rings]
    # carbon_chain_length = [rdFMCS.FindMCS([mol, carbon_mol]).smartsString if mol is not None else "" for mol in mols]
    carbon_chain_length = [smart.count('[#6]') if smart is not None else 0 for smart in carbon_chain_length]
    cot_list = [f" The longest carbon chain is {ccl} carbons long." for ccl in carbon_chain_length]
    
    return cot_list

def map_num_double_bond(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    bond_list = [mol.GetBonds() if mol is not None else [] for mol in mols]
    nums = [sum([b.GetBondType()==Chem.BondType.DOUBLE for b in list(bonds)]) for bonds in bond_list]
    cot_list = [f" The molecule has {num} double bonds." for num in nums]
    return cot_list

def map_chiral_center_cot(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    chiral_centers = [FindMolChiralCenters(mol) if mol is not None else [("","")] for mol in mols]
    r_count = [len([c for c in chiral if c[1] == 'R']) for chiral in chiral_centers]
    s_count = [len([c for c in chiral if c[1] == 'S']) for chiral in chiral_centers]
    cot_list = [f" The molecule has {rc+sc} chiral centers: {sc} with S configuration and {rc} with R configuration." for rc, sc in zip(r_count, s_count)]
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
    with open('resource/data/total_ring_to_iupac.json', 'r') as fp:
        ring_name_dict = json.load(fp)
    ring_info_multiset_iupac = [[ring_name_dict.get(smi, "unknown") for smi in smis] for smis in ring_smiles]
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

def map_scaffold_cot(smiles_list):
    scaffolds = [MurckoScaffoldSmilesFromSmiles(smiles) for smiles in smiles_list]
    with open('resource/data/scaffold_to_iupac.json', 'r') as fp:
        scaffold_name_dict = json.load(fp)
    scaffold_iupac = [scaffold_name_dict.get(smi, "unknown") for smi in scaffolds]
    cot_list = [f" The scaffold is {scaffold}." if len(scaffold)>0 else " The scaffold is unknown." for scaffold in scaffold_iupac]

    return cot_list


functional_group_smarts_dict = {
    'acid': '[!H0;F,Cl,Br,I,N+,$([OH]-*=[!#6]),+]',
    'acyl halide': '[#6X3;H0](=[OX1H0])([FX1,ClX1,BrX1,IX1])[!H]',
    'alcohol': '[#6][OX2H]',
    'aldehyde': '[CX3H1](=O)[#6]',
    'alkane': '[CX4]',
    'alkene': '[CX3]=[CX3]',
    'alkylaluminium': '[Al][C,c]',
    'alkyllithium': '[Li+;H0].[C-]',
    # 'alkylmagnesium halide': ('[I-,Br-,Cl-,F-].[Mg+][C,c]', '[I,Br,Cl,F][Mg]', '[c-,C-].[Mg+2].[I-,Br-,Cl-,F-]'),
    'alkyne': '[CX2]#C',
    # 'amine': ('[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]', '[ND3]([CX4])([CX4])[CX4]', '[NX3H0+0,NX4H1+;!$([N][!c]);!$([N]*~[#7,#8,#15,#16])]', '[NX3H0+0,NX4H1+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]', '[NX3H0+0,NX4H1+;$([N]([c])([C])[#6]);!$([N]*~[#7,#8,#15,#16])]', '[#6]-[#7](-[#6])-[#6]', '[CX4][NH2]', '[NX3H2+0,NX4H3+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]', '[NX3H2+0,NX4H3+]c', '[N+X4]([c,C])([c,C])([c,C])[c,C]'),
    # 'amide': ['[NX3][CX3](=[OX1])[#6]', 'O=C([c,CX4])[$([NH2]),$([NH][c,CX4]),$(N([c,CX4])[c,CX4])]', '[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]', '[*][CX3](=[OX1H0])[NX3]([*])([*])'],
    # 'amidine': ['[*][NX2]=[CX3H0]([*])[NX3]([*])([*])', '[NX2H1]=[CX3H0]([*])[NX3]([*])([*])', '[NX2H1]=[CX3H0]([*])[NX3H1]([*])', '[NX2H1]=[CX3H0]([*])[NX3H2]', '[*][NX2]=[CX3H0]([*])[NX3H1]([*])', '[*][NX2]=[CX3H0]([*])[NX3H2]'],
    'anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
    'aromatic': '[c]',
    'azide': '[NX2]=[N+X2H0]=[N-X1H0]',
    'azo': '[*][NX2]=[NX2][*]',
    'benzene': 'c1ccccc1',
    # 'borinic acid': ('[BX3;H0]([OX2H1])([!O])[!O]', '[BX3;H1]([OX2H1])[!O]', '[BX3;H2][OX2H1]'),
    'borinic ester': '[BX3;H0]([OX2H0])([!O])[!O]',
    'boronic acid': '[BX3]([OX2H])([OX2H])',
    'boronic ester': '[BX3;H0]([OX2H0])([OX2H0])[!O@!H]',
    'branched alkane': 'CC(C)C',
    'bromoalkane': '[#6][Br]',
    'carbamate': '[OX2][CX3H0](=[OX1H0])[NX3]',
    'carbodithio': '[#6X3;H0](=[SX1H0])([!H])[SX2H0]([!H])',
    'carbodithioic acid': '[#6X3;H0](=[SX1H0])([!H])[SX2H1]',
    'carbonate': '[!H][OX2H0][CX3H0](=[OX1H0])[OX2H0][!H]',
    'carbothioic o acid': '[#6X3;H0]([OX2H1])(=[SX1H0])([!H])',
    'carbothioic s acid': '[#6X3;H0](=[OX1H0])([SX2H1])([!H])',
    'carboxylate': '[C][C](=[OX1H0])[O-X1H0]',
    'carboxylic acid': '[CX3](=O)[OX2H1]',
    'carboxylic anhydride': '[*][CX3H0](=[OX1H0])[OX2H0][CX3H0](=[OX1H0])[*]',
    'chloroalkane': '[#6][Cl]',
    'cyanate': '[*][OX2H0][CX2H0]#[NX1H0]',
    'cyanide': 'C#N',
    'disulfide': '[#16X2H0][#16X2H0]',
    'ester': '[OX2H0][#6;!$([C]=[O])]',
    'ether': '[OD2]([#6])[#6]',
    'fluoroalkane': '[#6][F]',
    'haloalkane': '[#6][F,Cl,Br,I]',
    'hydroperoxide': '[!H][OX2H0][OX2H1]',
    'imide': '[CX3H0](=[OX1H0])([*])[NX3][CX3H0](=[OX1H0])[*]',
    'iodoalkane': '[#6][I]',
    'isocyanate': '[NX2H0]=[CX2H0]=[OX1H0]',
    'isonitrile': '[!H][NX2H0]=[CX2H0]=[SX1H0]',
    'isothiocyanate': '[!H][NX2H0]=[CX2H0]=[SX1H0]',
    'ketone': '[#6][CX3](=O)[#6]',
    'mercaptan': '[#16X2H]',
    'methylenedioxy': '[CX4H2;R]([OX2H0;R])([OX2H0;R])',
    'nitrate': '[OX2][N+X3H0](=[OX1H0])[O-X1H0]',
    'nitrile': '[NX1]#[CX2]',
    'nitrite': '[OX2][NX2H0]=[OX1H0]',
    'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    'nitroso': '[*][NX2]=[OX1H0]',
    # 'organic': ['[CX4]', '[CX3]=[CX3]', '[CX2]#C', '[c]', '[C][H]', '[C@H]', '[CR]', '[NX3][CX3](=[OX1])[#6]', 'O=C([c,CX4])[$([NH2]),$([NH][c,CX4]),$(N([c,CX4])[c,CX4])]', '[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]', '[*][CX3](=[OX1H0])[NX3]([*])([*])'],
    'orthocarbonate ester': '[CX4H0]([OX2])([OX2])([OX2])([OX2])',
    'orthoester': '[*][CX4]([OX2H0])([OX2H0])([OX2H0])',
    'oxime': '[!H][CX3]([*])=[NX2H0][OX2H1]',
    'peroxide': '[!H][OX2H0][OX2H0][!H]',
    'polyol': '[#6][OX2H]',
    'phenol': '[OX2H][cX3]:[c]',
    'phosphate': '[PX4;H0](=O)([OX2H])([OX2H])[OX2H0]',
    'phosphine': '[PX3]',
    'phosphodiester': '[PX4;H0](=O)([OX2H])([OX2H0])[OX2H0]',
    'phosphonic acid': '[PX4](=O)([OX2H])[OX2H]',
    'primary aldimine': '[*][CX3H1]=[NX2H1]',
    'primary amine': '[CX4][NH2]',
    'primary ketimine': '[*][CX3H0](=[NX2H1])([*])',
    'pyridyl': 'c1ccncc1',
    'quat': '[N+X4]([c,C])([c,C])([c,C])[c,C]',
    'secondary aldimine': '[*][CX3H1]=[NX2H0]',
    'secondary amine': '[$([NH]([CX4])[CX4]);!$([NH]([CX4])[CX4][O,N]);!$([NH]([CX4])[CX4][O,N])]',
    'secondary ketimine': '[*][CX3H0]([*])=[NX2H0]([*])',
    'siloxane': '[Si][O][Si]',
    'silyl ether': '[SiX4]([OX2H0])([!H])([!H])[!H]',
    'sulfide': '[!#16][#16X2H0][!#16]',
    'sulfinic acid': '[SX3H0](=O)([OX2H])[!H]',
    'sulfonate ester': '[SX4H0](=O)(=O)([OX2H0])[!H]',
    'sulfone': '[SX4H0](=O)(=O)([OX2H0])[!H]',
    'sulfonic acid': '[SX4H0](=O)(=O)([OX2H])[!H]',
    'sulfoxide': '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]',
    # 'tertiary amine': ['[ND3]([CX4])([CX4])[CX4]', '[NX3H0+0,NX4H1+;!$([N][!c]);!$([N]*~[#7,#8,#15,#16])]', '[NX3H0+0,NX4H1+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]',  '[NX3H0+0,NX4H1+;$([N]([c])([C])[#6]);!$([N]*~[#7,#8,#15,#16])]', '[#6]-[#7](-[#6])-[#6]'],
    'thial': '[#6X3;H1](=[SX1H0])([!H])',
    'thiocyanate': '[SX2H0]([!H])[CH0]#[NX1H0]',
    'thioketone': '[#6X3;H0]([!H])([!H])=[SX1H0]',
    'thiolester': '[#6X3;H0](=[OX1H0])([*])[SX2H0][!H]',
    'thionoester': '[#6X3;H0](=[SX1H0])([*])[OX2H0][!H]',
    # 'organic': ['[C][H]', '[C@H]', '[CR]']
}

def smarts_to_smiles(smarts):
    if isinstance(smarts, str):
        mol = Chem.MolFromSmarts(smarts)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return ""
        
    else:
        result = []
        for smart in smarts:
            mol = Chem.MolFromSmarts(smart)
            if mol is not None:
                result.append(Chem.MolToSmiles(mol))
            else:
                result.append("")
        return result

def zip_smiles_and_functional_group(groups, smiles):
    result = ""
    for group, smi in zip(groups, smiles):
        result += f" {group} group {smi},"
    
    if result[-1] == ",":
        result = result[:-1]
    
    result = result.replace(" ,", ",")
    return result.strip()

def map_functional_group_cot(smiles_list, mode='simple'):
    mols = [Chem.MolFromSmiles(target) for target in smiles_list]
    functional_group_list = getmembers(functional_groups, isfunction)
    functional_group_list = [(name, function) for name, function in functional_group_list if 'is' in name]
    functional_group_name_list = [' '.join(name.split('_')[1:]) for name, _ in functional_group_list]
    functional_group_smiles_dict = {name: smarts_to_smiles(smarts) for name, smarts in functional_group_smarts_dict.items()}
    
    mol_groups = []
    for mol in mols:
        if mol is None:
            mol_groups.append([])
            continue
        groups = [name for name, func in functional_group_list if func(mol)]
        groups = [group.split('_')[1] for group in groups]
        mol_groups.append(groups)
        
        
    if mode == 'simple':
        cot_list = [f" The functional groups present in the molecule include {', '.join(groups)} groups." if len(groups)>0 else " The functional group of the molecule is unknown." for groups in mol_groups]
    else:
        func_group_smiles = [[functional_group_smiles_dict.get(group, "") for group in groups] for groups in mol_groups]
        cot_list = [f" The functional groups present in the molecule include {zip_smiles_and_functional_group(groups, smis)}." if len(groups)>0 else " The functional group of the molecule is unknown." for groups, smis in zip(mol_groups, func_group_smiles)]
    
    cot_list = [','.join(cot.split(',')[:-1]) + ', and' +cot.split(',')[-1] if cot != ' The functional group of the molecule is unknown.' else cot for cot in cot_list]
    
    
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
    
    independent_rings = set(ri)-set(flatten(final_fused_set))-set(flatten(final_spiro_set))-set(flatten(final_bridge_set))
    substructures_independent = [Chem.rdmolfiles.MolFragmentToSmiles(mol, atomsToUse=atom_index) for atom_index in independent_rings]
    return substructures_spiro, substructures_fused, substructures_bridge, substructures_independent
        
def get_connected_ring_name(ri, mol, ring_name_dict):
    
    if len(ri) == 0:
        return []
    substructures_spiro, substructures_fused, substructures_bridge, substructure_independent = get_ring_substructure(ri, mol)
    substructures_spiro = [('spiro', s) for s in substructures_spiro]
    substructures_fused = [('fused', s) for s in substructures_fused]
    substructures_bridge = [('bridge', s) for s in substructures_bridge]
    substructure_independent = [('independent', s) for s in substructure_independent]
    result_substructures = substructures_fused + substructures_spiro + substructures_bridge + substructure_independent
    final_result = []
    for ring_type, smi in result_substructures:
        # ring_size = len("".join(filter(str.isalpha, smi)))
        iupac = ring_name_dict.get(smi, f"unknown {ring_type}")
        if iupac in ['unknown', '']:
            ring_mol = Chem.MolFromSmiles(smi)
            # (Connected) Ring molecule is not available
            if ring_mol is None:
                # print(Chem.MolToSmiles(mol))
                final_result.append(f"unknown {ring_type}")
                continue
            # Cannot map IUPAC name for the ring substructure -> try to decompose the ring further
            decomposed_ring_info = ring_mol.GetRingInfo().AtomRings()
            decomposed_rings = [Chem.rdmolfiles.MolFragmentToSmiles(ring_mol, atomsToUse=s) for s in decomposed_ring_info]
            decomposed_result_list = [ring_name_dict.get(sub_ring, "unknown independent") for sub_ring in decomposed_rings]
            decomposed_result_list = [r if r is not None else 'unknown independent' for r in decomposed_result_list]
            final_result.extend(decomposed_result_list)
        else:
            final_result.append(iupac)
                
    return final_result

def map_connected_ring_name_cot(smiles_list):
    mols = [Chem.MolFromSmiles(s) if type(s) == str else None for s in smiles_list]
    ring_info = [mol.GetRingInfo().AtomRings() if mol is not None else "" for mol in mols]
    with open('resource/data/total_ring_to_iupac.json', 'r') as fp:
        ring_name_dict = json.load(fp)
    ring_connectivity = [get_connected_ring_name(ri, mol, ring_name_dict) for ri, mol in zip(tqdm(ring_info), mols)]
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
            cot = " It does not include any ring."

        cot = cot.replace("  ", " ")
        if len(cot.split(', ')) > 1:
            cot = ', '.join(cot.split(', ')[:-1]) + ', and ' + cot.split(', ')[-1]
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
    try:
        cot_mode = hparams.cot_mode
    except:
        cot_mode = hparams['cot_mode']
    
    return cot_mode

def add_cot_to_target(examples, targets, cot_mode):
    cot_modes = cot_mode.split('-')
    cot_modes.reverse()
    for cm in cot_modes:
        if cm == '':
            break
        if cm not in TOTAL_COT_MODES:
            raise ValueError(f"Invalid CoT mode: {cm}")

        targets = [f"{cot}{target}" for target, cot in zip(targets, examples[f'cot_{cm}'])]

    
    return targets

def add_cot_to_text(examples, targets, direction='forward'):
    # direction: forward (caption2smiles), backward (smiles2caption)
    if direction == 'forward':
        targets = [f"{cot}{target}" for target, cot in zip(targets, examples['cot'])]
    else:
        targets = [f"{target}{cot}" for target, cot in zip(targets, examples['cot'])]
    return targets
    
def map_cot_to_smiles_list(smiles_list, hparams, data_dict, split):
    run_name = map_cot_mode(hparams)
    cot_list_total = ["" for _ in range(len(smiles_list))]
    cot_modes = run_name.split('-')
    
    for cm in cot_modes:
        if cm == '':
            break
        
        if cm not in TOTAL_COT_MODES:

            raise ValueError(f"Invalid CoT mode: {cm}")

        cot_function_dict = {'func_simple': map_functional_group_cot, 'func_smiles': map_functional_group_cot, 'scaffold': map_scaffold_cot, \
                            'chain': map_carbon_chain_length, 'fragment': map_fragment_cot, 'ring': map_ring_cot, 'multiset_simple': map_multiset_cot, \
                            'multiset_full': map_multiset_cot, 'multiset_formula': map_multiset_cot, 'multiset_type': map_multiset_cot, \
                            'aromatic': map_aromatic_ring_cot, 'ring_name': map_ring_name_cot, 'con_ring_name': map_connected_ring_name_cot, \
                            'iupac': map_iupac_cot, 'double_bond': map_num_double_bond, 'chiral': map_chiral_center_cot,
                            'weight': map_weight_cot, 'name': map_name_cot, 'func_chem': map_chem_functional_group_cot
                            }
        cot_function = cot_function_dict.get(cm)
        if ('multiset' in cm) or ('func' in cm):
            mode = cm.split('_')[1]
            cot_list = cot_function(smiles_list, mode=mode)
        else:
            cot_list = cot_function(smiles_list)
        
        data_dict[f'cot_{cm}'] = cot_list
    
    
    cot_list = add_cot_to_target(data_dict, cot_list_total, run_name)
    data_dict['cot'] = cot_list
    
    return data_dict

# Codes adapted from https://github.com/ur-whitelab/chemcrow-public

def is_cas(text):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None

def smiles2name(smi, single_name=True):
    """This function queries the given molecule smiles and returns a name record or iupac"""

    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except Exception:
        raise ValueError("Invalid SMILES string")
    # query the PubChem database
    r = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        + smi
        + "/synonyms/JSON"
    )
    # return the SMILES string
    try:
        data = r.json()
        if single_name:
            index = 0
            names = data["InformationList"]["Information"][0]["Synonym"]
            while is_cas(name := names[index]):
                index += 1
                if index == len(names):
                    raise ValueError("No name found")
        else:
            name = data["InformationList"]["Information"][0]["Synonym"]
    except:
        name = ""
    return name

def map_name_cot(smiles_list):
    iupac_path = os.path.join('ChEBI-20_data', f'dict_iupac.json')
    iupac_dict = json.load(open(iupac_path, 'r'))
    iupac_list = [iupac_dict.get(smi, "") for smi in smiles_list]
    cot_list = [f" The IUPAC name of the molecule is {iupac}." if len(iupac)>0 else " The name of the molecule is not available." for iupac in iupac_list]
    
    return cot_list

def smiles2weight(smi):
    mol = Chem.MolFromSmiles(smi)
    return rdMolDescriptors.CalcExactMolWt(mol)

def map_weight_cot(smiles_list):
    weight_list = [smiles2weight(smi) for smi in tqdm(smiles_list)]
    cot_list = [f" The molecular weight is {round(weight,2)}g/mol." for weight in weight_list]
    
    return cot_list

def map_chem_functional_group_cot(smiles_list): 
    fg_dict = {
            "furan": "o1cccc1",
            "aldehydes": " [CX3H1](=O)[#6]",
            "esters": " [#6][CX3](=O)[OX2H0][#6]",
            "ketones": " [#6][CX3](=O)[#6]",
            "amides": " C(=O)-N",
            "thiol groups": " [SH]",
            "alcohol groups": " [OH]",
            "methylamide": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
            "carboxylic acids": "*-C(=O)[O;D1]",
            "carbonyl methylester": "*-C(=O)[O;D2]-[C;D1;H3]",
            "terminal aldehyde": "*-C(=O)-[C;D1]",
            "amide": "*-C(=O)-[N;D1]",
            "carbonyl methyl": "*-C(=O)-[C;D1;H3]",
            "isocyanate": "*-[N;D2]=[C;D2]=[O;D1]",
            "isothiocyanate": "*-[N;D2]=[C;D2]=[S;D1]",
            "nitro": "*-[N;D3](=[O;D1])[O;D1]",
            "nitroso": "*-[N;R0]=[O;D1]",
            "oximes": "*=[N;R0]-[O;D1]",
            "Imines": "*-[N;R0]=[C;D1;H2]",
            "terminal azo": "*-[N;D2]=[N;D2]-[C;D1;H3]",
            "hydrazines": "*-[N;D2]=[N;D1]",
            "diazo": "*-[N;D2]#[N;D1]",
            "cyano": "*-[C;D2]#[N;D1]",
            "primary sulfonamide": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
            "methyl sulfonamide": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
            "sulfonic acid": "*-[S;D4](=O)(=O)-[O;D1]",
            "methyl ester sulfonyl": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
            "methyl sulfonyl": "*-[S;D4](=O)(=O)-[C;D1;H3]",
            "sulfonyl chloride": "*-[S;D4](=O)(=O)-[Cl]",
            "methyl sulfinyl": "*-[S;D3](=O)-[C;D1]",
            "methyl thio": "*-[S;D2]-[C;D1;H3]",
            "thiols": "*-[S;D1]",
            "thio carbonyls": "*=[S;D1]",
            "halogens": "*-[#9,#17,#35,#53]",
            "t-butyl": "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
            "tri fluoromethyl": "*-[C;D4](F)(F)F",
            "acetylenes": "*-[C;D2]#[C;D1;H]",
            "cyclopropyl": "*-[C;D3]1-[C;D2]-[C;D2]1",
            "ethoxy": "*-[O;D2]-[C;D2]-[C;D1;H3]",
            "methoxy": "*-[O;D2]-[C;D1;H3]",
            "side-chain hydroxyls": "*-[O;D1]",
            "ketones": "*=[O;D1]",
            "primary amines": "*-[N;D1]",
            "nitriles": "*#[N;D1]",
        }
    
    fgmol_dict = {fg_name: Chem.MolFromSmarts(fg) for fg_name, fg in fg_dict.items()}
    fg_list = [smiles2chem_functional_group(smi, fgmol_dict) for smi in smiles_list]
    fg_cot = []
    for fgs_in_molec in fg_list:
        if len(fgs_in_molec) > 1:
            fg_cot.append(f"This molecule contains {', '.join(fgs_in_molec[:-1])}, and {fgs_in_molec[-1]}.")
        elif len(fgs_in_molec) == 1:
            fg_cot.append(f"This molecule contains {fgs_in_molec[0]}.")
        else:
            fg_cot.append("This molecule does not contain any functional group.")
    return fg_cot

    
def smiles2chem_functional_group(smi, fgmol_dict):
    mol = Chem.MolFromSmiles(smi)
    fg_list = []
    for fg_name, fg_mol in fgmol_dict.items():
        if len(Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True))>0:
            fg_list.append(fg_name)
    return fg_list
        