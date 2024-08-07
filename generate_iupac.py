import pandas as pd
from analysis import compute_cot_accuracy, map_ring_size_from_cot
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm
import re
import json
from util_cot import map_iupac_cot, smiles_to_iupac

from itertools import combinations
from util_cot import merge_connected_ring, get_ring_substructure, get_subs

print(os.getcwd())


def generate_molecule_iupac(smiles_list):
    iupac_list = [smiles_to_iupac(smi) for smi in tqdm(smiles_list)]
    output_path = os.path.join('ChEBI-20_data', f'{split}_iupac.txt')

    with open(output_path, 'w') as f:
        for frag in iupac_list:
            f.write(frag)
            f.write('\n')

def generate_connect_ring_iupac(ri, mol):
    
    if len(ri) == 0:
        return []
    
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
    
    result_substructures = substructures_fused + substructures_spiro + substructures_bridge
    final_result = []
    for smi in result_substructures:
        final_result.append(smi)
        sub_mol = Chem.MolFromSmiles(smi)
        if sub_mol is None:
            continue
        sub_ring_info = sub_mol.GetRingInfo().AtomRings()
        sub_rings = [Chem.rdmolfiles.MolFragmentToSmiles(sub_mol, atomsToUse=s) for s in sub_ring_info]
        final_result.extend(sub_rings)

    return final_result

def flatten(xss):
    return [x for xs in xss for x in xs]

total_smiles_list = []
for split in ['train', 'test']:
    smiles_list_path = os.path.join('ChEBI-20_data', f'{split}.txt')
    smiles_pair_list = [
    [pair.split()[0], pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
    ][1:]
    smiles_list = [pair[1] for pair in smiles_pair_list]
    total_smiles_list.extend(smiles_list)
    
mols = [Chem.MolFromSmiles(s) for s in total_smiles_list]
ring_info = [mol.GetRingInfo().AtomRings() for mol in mols]
connected_rings = [generate_connect_ring_iupac(ri, mol) for ri, mol in zip(tqdm(ring_info), mols)]
ring_smiles = flatten(connected_rings)
ring_smiles = set(ring_smiles)

connected_ring_iupac_dict = {}
for smi in tqdm(ring_smiles, 'Map IUPAC'):
    connected_ring_iupac_dict[smi] = smiles_to_iupac(smi)

with open('resource/data/total_ring_to_iupac.json', 'w') as f:
    json.dump(connected_ring_iupac_dict, f)





