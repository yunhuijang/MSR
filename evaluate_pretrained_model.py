import os
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import argparse
import wandb
from os.path import join

from util_cot import map_ring_cot, map_multiset_cot
from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric

os.environ["WANDB__SERVICE_WAIT"] = "300"

def predict_with_cot(hparams):
    architecture = hparams.architecture
    cot_mode_multiset = hparams.cot_mode_multiset
    cot_mode_ring = hparams.cot_mode_ring
    cot_mode_fragment = hparams.cot_mode_fragment
    split = hparams.split
    batch_size_generate = hparams.batch_size_generate
    task = hparams.finetune_task

    tokenizer = T5Tokenizer.from_pretrained(f"laituan245/{architecture}{task}", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(f'laituan245/{architecture}{task}')
    
    device = 'cuda:0'
    model.to(device)
    
    if split == 'test':
        if task == "":
            smiles_list_path = os.path.join('predictions', f"{architecture}-caption2smiles.txt")
        else:
            smiles_list_path = os.path.join('predictions', f"{architecture}{task}.txt")
        smiles_pair_list = [
        [" ".join(pair.split()[:-2]), pair.split()[-2], pair.split()[-1]] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        ][1:]
        description_list = [pair[0] for pair in smiles_pair_list]
        gt_smiles_list = [pair[1] for pair in smiles_pair_list]

    elif split == 'train':
        smiles_list_path = os.path.join('ChEBI-20_data', 'train.txt')
        smiles_pair_list = [
        [" ".join(pair.split()[0]), pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        ][1:][:3301] # 3301
        description_list = [pair[2] for pair in smiles_pair_list]
        gt_smiles_list = [pair[1] for pair in smiles_pair_list]
    
    
    ring_cot_list = map_ring_cot(gt_smiles_list)
    multiset_cot_list = map_multiset_cot(gt_smiles_list, mode=cot_mode_multiset)
    

    run_name = ""
    if cot_mode_multiset in ['simple', 'full']:
        run_name += f'-multiset_{cot_mode_multiset}'
    if cot_mode_ring:
        run_name += '-ring'
    if cot_mode_fragment:
        run_name += '-frag'


    prediction_list= []
    
    for index in tqdm(range(len(prediction_list), len(description_list), batch_size_generate)):
        input_text = description_list[index:index+batch_size_generate]
        if cot_mode_ring:
            input_text = [it+rc for it, rc in zip(input_text, ring_cot_list[index:index+batch_size_generate])]
        if cot_mode_multiset in ['simple', 'full']:
            input_text = [it+rc for it, rc in zip(input_text, multiset_cot_list[index:index+batch_size_generate])]
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
        outputs = model.generate(input_ids.to(device), num_beams=5, max_length=512)
        prediction = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        prediction_list.extend(prediction)

    if split == 'test':
        file_name = f'predictions/cot/{architecture}{task}{run_name}'
        
    elif split == 'train':
        file_name = f'predictions/cot/{architecture}{task}{run_name}-{split}'
    
    with open(f'{file_name}.txt', 'w') as f:
        f.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
        for desc, rt, ot in zip(description_list, gt_smiles_list, prediction_list):
            f.write(desc + '\t' + rt + '\t' + ot + '\n')
            
    return prediction_list

def evaluate(architecture, task, run_name, split='test'):
    if split == 'test':
        file_name = f'{architecture}{task}{run_name}.txt'
        
    elif split == 'train':
        file_name = f'{architecture}{task}{run_name}-{split}.txt'
    # file_name = f'{architecture}-{task}{run_name}.txt'
    file_path = join('predictions', 'cot', file_name)
    
    smiles_list_path = os.path.join(file_path)
    smiles_pair_list = [
    [" ".join(pair.split()[:-2]), pair.split()[-2], pair.split()[-1]] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
    ][1:]

    table = wandb.Table(data=smiles_pair_list,
                        columns=['description', 'gt_smiles', 'predicted_smiles'])
    
    wandb.log({f"Prediction": table})
    
    bleu_score, exact_match_score, levenshtein_score, validity_score = mol_translation_metrics.evaluate(file_path)
    validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fingerprint_metrics.evaluate(file_path, 2)
    fcd_metric_score = fcd_metric.evaluate(file_path)
    # wandb.log(f'For {file_name}\n')
    wandb.log({"BLEU": round(bleu_score, 3), "Exact": round(exact_match_score, 3),
               "Levenshtein": round(levenshtein_score, 3), "MACCS FTS": round(maccs_sims_score, 3),
                "RDK FTS": round(rdk_sims_score, 3), "Morgan FTS": round(morgan_sims_score, 3),
                "FCD Metric": round(fcd_metric_score, 3), "Validity": round(validity_score, 3)
               })

@staticmethod
def add_args(parser):
    parser.add_argument("--architecture", type=str, default='molt5-small')
    parser.add_argument("--cot_mode_multiset", type=str, default='')
    parser.add_argument("--cot_mode_fragment", action='store_true')
    parser.add_argument("--cot_mode_ring", action='store_true')
    parser.add_argument("--wandb_mode", type=str, default='online')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--batch_size_generate", type=int, default=16)
    parser.add_argument("--finetune_task", type=str, default='-caption2smiles')


    return parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    hparams = parser.parse_args()
    run_name = ""
    if hparams.cot_mode_multiset in ['simple', 'full']:
        run_name += f'-multiset({hparams.cot_mode_multiset})'
    if hparams.cot_mode_ring:
        run_name += '-ring'
    if hparams.cot_mode_fragment:
        run_name += '-frag'
    wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}', mode=hparams.wandb_mode)
    wandb.config.update(hparams)
    
    predict_with_cot(hparams)
    
    evaluate(hparams.architecture, hparams.finetune_task, run_name, hparams.split)
    
    wandb.finish()
    
    