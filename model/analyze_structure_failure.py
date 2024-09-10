from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from pathlib import Path
import random
from tqdm import tqdm
import argparse
import json
import openai
from openai import OpenAI
from huggingface_hub.hf_api import HfFolder
from inspect import getmembers, isfunction
from thermo import functional_groups

from util_cot import map_cot_mode, map_cot_to_smiles_list
from evaluation import text_translation_metrics, mol_translation_metrics, fingerprint_metrics, fcd_metric
import wandb
from analysis import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai_key = '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'
os.environ['OPENAI_API_KEY'] = openai_key   


def analyze_structure_failure(hparams):

    
    model_id = hparams.model_id
    max_length = hparams.max_length

    smiles_list_path = os.path.join('ChEBI-20_data', f'test.txt')
    smiles_pair_list = [
    [pair.split()[0], pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
    ][1:]
    gt_smiles_list_test = [pair[1] for pair in smiles_pair_list]    

    final_results = []

    # Set models
    if 'llama' in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
    elif 'gpt' in model_id:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))   
        
    functional_group_list = getmembers(functional_groups, isfunction)
    functional_group_list = [name.split('_')[1] for name, _ in functional_group_list if 'is' in name]
    functional_group_str = str(functional_group_list)[1:-1].replace('\'', '')
    
    
    for index, smiles in enumerate(tqdm(gt_smiles_list_test)):

        head_prompt =  f"You are now working as an excellent expert in chemisrty and drug discovery. \
                Given the SMILES representation of a molecule and structural description of the molecule, your job is to predict the structural information of the molecule. \
                The structural information of the molecule caption includes ALL the functional groups, the length of the longest carbon chain, the number of aromatic rings, and the IUPAC names of ALL the rings in the molecule. \
                The functional group and ring IUPAC name should be in the list. \
                Note that the functional groups should be in the candidates: {functional_group_str}\n" \
                + "\n"
        
        head_prompt += "Your response should only be in the JSON format following {\"functional_group\": , \"longest_carbon_chain_length\": , \"aromatic_ring\": , \"ring_IUPAC_name\":}; \
            THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. DO NOT CHANGE THE JSON KEY NAMES. "
        input_prompt = f"Input: {smiles}\n"
        
        messages = [
        {"role": "system", "content": f"{head_prompt}"},
        {"role": "user", "content": f"{input_prompt}"},
        ]

        if 'llama' in model_id:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=max_length,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            result = tokenizer.decode(response, skip_special_tokens=True)
            output = result

        
        elif 'gpt' in model_id:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                        {"role": "system", "content": head_prompt},
                        {"role": "user", "content": input_prompt},
                    ]
                )
            
            output = response.choices[0].message.content

        # filter only results (remove 'caption' tag)
        if output.find('}') == -1:
            output += '}'
        
        try:
            start_index = output.find('{')
            end_index = output.find('}')
            result = output[start_index:end_index+1]
            result = json.loads(result)
            if isinstance(result['functional_group'], list):
                result['functional_group'] = sorted(set(result['functional_group']))
            if isinstance(result['ring_IUPAC_name'], list):
                result['ring_IUPAC_name'] = sorted(set(result['ring_IUPAC_name']))
            
        except:
            if '}' in output and '{' in output:
                # some character exists after }
                result = output[output.find('{'):output.find('}')]
            else:
                result = output.strip().replace('\n', '')

            result = result.replace('\n', ' ')
            result = result.replace('\t', ' ')
        final_results.append(result)
    
    return gt_smiles_list_test, final_results

def cot_to_dict(cot):
    
    
    result_dict = {}
    cots = cot.split('.')
    cots[0]
    
    return result_dict


@staticmethod
def add_args(parser):
    parser.add_argument("--cot_mode_multiset", type=str, default='None')
    parser.add_argument("--cot_mode_fragment", action='store_true')
    parser.add_argument("--cot_mode_ring", action='store_true')
    parser.add_argument("--cot_mode_ring_name", action='store_true')
    parser.add_argument("--cot_mode_iupac", action='store_true')
    parser.add_argument("--cot_mode_scaffold", action='store_true')
    
    parser.add_argument("--cot_mode_aromatic", action='store_false')
    parser.add_argument("--cot_mode_chain", action='store_false')
    parser.add_argument("--cot_mode_con_ring_name", action='store_false')
    parser.add_argument("--cot_mode_functional_group", action='store_false')
    parser.add_argument("--functional_group_type", type=str, default='simple')

    parser.add_argument("--wandb_mode", type=str, default='disabled')

    parser.add_argument("--model_id", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', 
                        choices=['meta-llama/Meta-Llama-3-8B-Instruct', 'gpt-4o',
                                'meta-llama/Meta-Llama-3.1-70B-Instruct', 'meta-llama/Meta-Llama-3.1-405B-Instruct',
                                'meta-llama/Meta-Llama-3.1-8B-Instruct'])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--architecture", type=str, default='llama3', choices=['llama3', 'gpt4o'])

    return parser

def evaluate_failure(file_name, gt_smiles_list_test, final_results):
    # with open(f'{file_name}', 'w') as f:
    #     f.write('description' + '\t' + 'output' + '\n')
    #     for rt, ot in zip(gt_smiles_list_test, final_results):
    #         f.write(rt + '\t' + str(ot) + '\n')
               
    cot_list_chain = map_cot_to_smiles_list(gt_smiles_list_test, hparams, {}, 'test')['cot_chain']
    chain_info_list = [map_chain_from_cot(cot) for cot in cot_list_chain]
    
    cot_list_arom = map_cot_to_smiles_list(gt_smiles_list_test, hparams, {}, 'test')['cot_aromatic']
    arom_info_list = [map_arom_num_from_cot(cot) for cot in cot_list_arom]
    
    cot_list_func = map_cot_to_smiles_list(gt_smiles_list_test, hparams, {}, 'test')['cot_functional_group']
    func_info_list = [map_functional_group_from_cot(cot) for cot in cot_list_func]
    
    cot_list_con_ring_name = map_cot_to_smiles_list(gt_smiles_list_test, hparams, {}, 'test')['cot_connected_ring_name']
    ring_info_list = [map_ring_name_from_cot(cot) for cot in cot_list_con_ring_name]
    ring_info_list = [sorted(list(ri.keys())) for ri in ring_info_list]
    
    chain_match, arom_match, func_match, ring_match = 0, 0, 0, 0
    
    
    for pred_info, chain_info, arom_info, func_info, ring_info in zip(final_results, chain_info_list, arom_info_list, func_info_list, ring_info_list):
        pred_info = pred_info.replace('\'', '"')
        if '}' not in pred_info:
            pred_info += '}'
        try:
            pred_info = json.loads(pred_info)
        except:
            print(pred_info)
            continue
            
                  
        if pred_info['longest_carbon_chain_length'] == chain_info:
            chain_match += 1
        if pred_info['aromatic_ring'] == arom_info:
            arom_match += 1
        # if pred_info['functional_group'] == func_info:
        func_match += len(set(pred_info['functional_group']).intersection(set(func_info)))/len(func_info)
        if pred_info['ring_IUPAC_name'] == ring_info:
            ring_match += 1
    
    result = {'chain_match': chain_match/len(final_results), 'arom_match': arom_match/len(final_results),
              'func_match': func_match/len(final_results), 'ring_match': ring_match/len(final_results)}
    
    wandb.log(result)
    print(result)


if __name__ == "__main__":
    # for hugging face login
    HfFolder.save_token('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')
    
    parser = argparse.ArgumentParser()
    add_args(parser)
    hparams = parser.parse_args()
    cot_mode = map_cot_mode(hparams)
    
    file_name = f'predictions/generalist/{hparams.architecture}{cot_mode}-analysis.txt'
    # wandb.init(project='analysis', name=f'{hparams.architecture}-analysis',
    #             group='generalist', mode=hparams.wandb_mode)
    # wandb.config.update(hparams, allow_val_change=True)
    # gt_smiles_list_test, final_results = analyze_structure_failure(hparams)
    smiles_pair_list = [
    [pair.split('\t')[0], pair.split('\t')[1]] for pair in Path(file_name).read_text(encoding="utf-8").splitlines()
    ][1:]
    gt_smiles_list_test = [pair[0] for pair in smiles_pair_list]
    final_results = [pair[1] for pair in smiles_pair_list]

            
        
    evaluate_failure(file_name, gt_smiles_list_test, final_results)
                    