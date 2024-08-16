import json
import os
from pathlib import Path
import selfies
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles, GetScaffoldForMol 
from util_cot import smiles_to_iupac, canonicalize
from generate_iupac import generate_molecule_iupac
from tqdm import tqdm
from rdkit.Chem import rdFMCS
from util_cot import map_scaffold_cot, map_functional_group_cot
from os import listdir
from shutil import rmtree
# with open('ChEBI-20_data/task1_chebi20_text2mol_train.json', 'r') as f:
#     data = json.load(f)
# description_list_json = [d['input'] for d in data['Instances']]
# selfies_list_json = [d['output'] for d in data['Instances']]

# total_smiles_list = []
# for split in ['test', 'train', 'validation']:
#     smiles_list_path = os.path.join('ChEBI-20_data', f'{split}.txt')
#     smiles_pair_list = [
#     [pair.split()[0], pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
#     ][1:][:50]
#     smiles_list = [pair[1] for pair in smiles_pair_list]
#     total_smiles_list.extend(smiles_list)


# total_smiles_list = ['CN\\1C2=CC=CC=C2O/C1=C\\C3=CC=[N+](C4=CC=CC=C34)CCC[N+](C)(C)C']

# for i, mol in enumerate(mols):
#     try:
#         selfies.encoder(Chem.MolToSmiles(mol))
#     except:
#         print(i)
#         print(Chem.MolToSmiles(mol))
# sfs = [selfies.encoder(Chem.MolToSmiles(mol)) for mol in mols]
# data['instances']
print('hi')

dir_list = listdir('output')
for dir in dir_list:
    file_path = sorted([dI for dI in os.listdir(f'output/{dir}') if os.path.isdir(os.path.join(f'output/{dir}',dI))])
    if len(file_path) == 0:
        rmtree(f'output/{dir}')
    elif len(file_path) > 2:
       rmtree(f'output/{dir}/{file_path[1]}')
    
