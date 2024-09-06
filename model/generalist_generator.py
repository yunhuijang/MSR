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


from util_cot import map_cot_mode, map_cot_to_smiles_list
from evaluation import text_translation_metrics, mol_translation_metrics, fingerprint_metrics, fcd_metric
import wandb

openai_key = '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'
os.environ['OPENAI_API_KEY'] = openai_key   


def generalist(hparams):

    
    task = hparams.task
    k = hparams.k
    model_id = hparams.model_id
    max_length = hparams.max_length

    # load train data for examples (few-shot learning)
    smiles_list_path = os.path.join('ChEBI-20_data', f'train.txt')
    smiles_pair_list_train = [
    [pair.split()[0], pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
    ][1:]
    smiles_list_train = [pair[1] for pair in smiles_pair_list_train]
    cot_list_train = map_cot_to_smiles_list(smiles_list_train, hparams, {}, 'train')
    smiles_pair_list_train = [sm + [cot] for sm, cot in zip(smiles_pair_list_train, cot_list_train['cot'])]

    # load test data
    smiles_list_path = os.path.join('ChEBI-20_data', f'test.txt')
    smiles_pair_list = [
    [pair.split()[0], pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
    ][1:]
    gt_smiles_list_test = [pair[1] for pair in smiles_pair_list]
    cot_list_test = map_cot_to_smiles_list(gt_smiles_list_test, hparams, {}, 'test')['cot']
    description_list_test = [pair[2] for pair in smiles_pair_list]
    

    
    final_results = []
    wrong_results = []
    for index, (description, smiles, cot) in enumerate(zip(description_list_test, tqdm(gt_smiles_list_test), cot_list_test)):
        # sample k-shot data
        random.seed(index)
        k_shot_data = random.choices(smiles_pair_list_train, k=k)
        k_shot_description_list = [pair[2] for pair in k_shot_data]
        k_shot_smiles_list = [pair[1] for pair in k_shot_data]
        k_shot_cot_list = [pair[3] for pair in k_shot_data]
        
        # head / input prompt
        if task == 'text2mol':
            head_prompt = "You are now working as an excellent expert in chemisrty and drug discovery. \
                Given the caption of a molecule, your job is to predict the SMILES representation of the molecule. \
                The molecule caption is a sentence that describes the molecule, which mainly describes the molecule's structures, properties, and production. \
                You can infer the molecule SMILES representation from the caption.\n" \
                + "\n"
            for k_index, (k_des, k_smi, k_cot) in enumerate(zip(k_shot_description_list, k_shot_smiles_list, k_shot_cot_list)):
                head_prompt += f"Example {k_index+1}: \n" \
                    + "```\n" \
                    + f"Instruction: Given the caption of a molecule, predict the SMILES representation of the molecule.\n" \
                    + f"Input: {k_des}{k_cot}\n" \
                    + "```\n" \
                    + "\n" \
                    + "Your output should be: \n" \
                    + "```\n" \
                    + f"{{\"molecule\": \"{k_smi}\"}}" \
                    + "```\n" \
                    + "\n"
                        
            # head_prompt += "Your response should only be in the JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. "
            head_prompt += "Your response should include the molecule caption and the JSON format above. "
            input_prompt = f"Input: {description}"
            
        elif task == 'mol2text':
            head_prompt = "You are now working as an excellent expert in chemisrty and drug discovery. \
                Given the SMILES representation of a molecule and structural description of the molecule, your job is to predict the caption of the molecule. \
                The molecule caption is a sentence that describes the molecule, which mainly describes the molecule's structures, properties, and production.\n" \
                + "\n"
            for k_index, (k_des, k_smi, k_cot) in enumerate(zip(k_shot_description_list, k_shot_smiles_list, k_shot_cot_list)):
                head_prompt += f"Example {k_index+1}: \n" \
                    + "```\n" \
                    + f"Instruction: Given the SMILES representation of a molecule, predict the caption of the molecule.\n" \
                    + f"Input: {k_smi}{k_cot}\n" \
                    + f"" \
                    + "```\n" \
                    + "\n" \
                    + "Your output should be: \n" \
                    + "```\n" \
                    + f"{{\"caption\": \"{k_des}\"}}" \
                    + "```\n" \
                    + "\n"
            head_prompt += "Your response should only be in the JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. "
            input_prompt = f"Input: {smiles}{cot}\n"
        
        messages = [
        {"role": "system", "content": f"{head_prompt}"},
        {"role": "user", "content": f"{input_prompt}"},
        ]

        if 'llama' in model_id:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        
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
            
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))   
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                        {"role": "system", "content": head_prompt},
                        {"role": "user", "content": input_prompt},
                    ]
                )
            
            output = response.choices[0].message.content
            
        # filter only results (remove 'caption' tag)
        try:
            start_index = output.find('{')
            end_index = output.find('}')
            result = output[start_index:end_index+1]
            result = json.loads(result)
            result_smiles = list(result.values())[0]
        except:
            if 'caption' in output:
                if '}' in output:
                    # some character exists after }
                    result_smiles = output[output.find('caption')+len('caption')+4:result.find('}')-1]
                else:
                    # unclosed }
                    result_smiles = output[output.find('caption')+len('caption')+4:]
            else:
                # generates no 'caption'
                result_smiles = output
        
        final_results.append(result_smiles)
    
    
    return description_list_test, gt_smiles_list_test, final_results

def evaluate_mol2text(file_name, description_list_test, gt_smiles_list_test, final_results):
    with open(f'{file_name}', 'w') as f:
        f.write('SMILES' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
        for desc, rt, ot in zip(gt_smiles_list_test, description_list_test, final_results):
            f.write(desc + '\t' + rt + '\t' + ot + '\n')
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
            text_translation_metrics.evaluate(
                'allenai/scibert_scivocab_uncased', file_name, 512
            )
    result = {"BLEU2": round(bleu2, 3), "BLEU4": round(bleu4, 3),
            "ROUGE1": round(rouge_1, 3), "ROUGE2": round(rouge_2, 3),
            "ROUGEL": round(rouge_l, 3), "METEOR": round(meteor_score, 3)
            }
    wandb.log(result)

def evaluate_text2mol(file_name, description_list_test, gt_smiles_list_test, final_results):
    with open(f'{file_name}', 'w') as f:
            f.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
            for desc, rt, ot in zip(description_list_test, gt_smiles_list_test, final_results):
                f.write(desc + '\t' + rt + '\t' + ot + '\n')
                
    bleu_score, exact_match_score, levenshtein_score, validity_score = mol_translation_metrics.evaluate(file_name)
    validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fingerprint_metrics.evaluate(file_name, 2)
    fcd_metric_score = fcd_metric.evaluate(file_name)
    result = {"BLEU": round(bleu_score, 3), "Exact": round(exact_match_score, 3),
            "Levenshtein": round(levenshtein_score, 3), "MACCS FTS": round(maccs_sims_score, 3),
            "RDK FTS": round(rdk_sims_score, 3), "Morgan FTS": round(morgan_sims_score, 3),
            "FCD Metric": round(fcd_metric_score, 3), "Validity": round(validity_score, 3)
            }
    wandb.log(result)


@staticmethod
def add_args(parser):
    parser.add_argument("--cot_mode_multiset", type=str, default='None')
    parser.add_argument("--cot_mode_fragment", action='store_true')
    parser.add_argument("--cot_mode_ring", action='store_true')
    parser.add_argument("--cot_mode_ring_name", action='store_true')
    parser.add_argument("--cot_mode_iupac", action='store_true')
    parser.add_argument("--cot_mode_scaffold", action='store_true')
    
    parser.add_argument("--cot_mode_aromatic", action='store_true')
    parser.add_argument("--cot_mode_chain", action='store_true')
    parser.add_argument("--cot_mode_con_ring_name", action='store_true')
    parser.add_argument("--cot_mode_functional_group", action='store_true')

    parser.add_argument("--wandb_mode", type=str, default='disabled')

    parser.add_argument("--task", type=str, default='mol2text', choices=['mol2text', 'text2mol'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model_id", type=str, default='meta-llama/Meta-Llama-3.1-405B-Instruct', 
                        choices=['meta-llama/Meta-Llama-3-8B-Instruct', 'gpt-4o',
                                'meta-llama/Meta-Llama-3.1-70B-Instruct', 'meta-llama/Meta-Llama-3.1-405B-Instruct',
                                'meta-llama/Meta-Llama-3.1-8B-Instruct'])
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--architecture", type=str, default='llama3', choices=['llama3', 'gpt4o'])

    return parser

if __name__ == "__main__":
    # for hugging face login
    HfFolder.save_token('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')
    
    parser = argparse.ArgumentParser()
    add_args(parser)
    hparams = parser.parse_args()
    cot_mode = map_cot_mode(hparams)
    
    file_name = f'predictions/generalist/{hparams.architecture}-{hparams.task}{cot_mode}-{hparams.k}.txt'
    wandb.init(project=hparams.task.split('2')[1]+'2'+hparams.task.split('2')[0], name=f'{hparams.architecture}-{hparams.task}{cot_mode}-{hparams.k}',
                group='generalist', mode=hparams.wandb_mode)

    description_list_test, gt_smiles_list_test, final_results = generalist(hparams)
    if hparams.task == 'mol2text':
        evaluate_mol2text(file_name, description_list_test, gt_smiles_list_test, final_results)
    elif hparams.task == 'text2mol':
        evaluate_text2mol(file_name, description_list_test, gt_smiles_list_test, final_results)
                    