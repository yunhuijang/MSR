import numpy as np
from tqdm import tqdm
from rdkit import Chem
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["WANDB__SERVICE_WAIT"] = "300"

import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.integrations import WandbCallback

from pathlib import Path
import wandb
import pytorch_lightning as pl
from huggingface_hub.hf_api import HfFolder
import torch
import selfies

from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric
from util_cot import map_ring_cot, map_multiset_cot, map_fragment_cot, map_cot_mode, add_cot_to_target, map_aromatic_ring_cot, map_carbon_chain_length, map_ring_name_cot, map_iupac_cot, map_connected_ring_name_cot
from analysis import compute_cot_accuracy

class FineTuneTranslator(pl.LightningModule):
    def __init__(self, hparams):
        super(FineTuneTranslator, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_model(hparams)
        self.setup_datasets(hparams)        
        self.sanity_checked = False
    
    def load_dataset(self, split):
        # <FIX> Need to be fixed when CoT added
        smiles_list_path = os.path.join('ChEBI-20_data', f'{split}.txt')
        smiles_pair_list = [
        [pair.split()[0], pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
        ][1:]
        # if self.hparams.test:
        #     smiles_pair_list = smiles_pair_list[:20]
        description_list = [pair[2] for pair in smiles_pair_list]
        gt_smiles_list = [pair[1] for pair in smiles_pair_list]
        id_list = [pair[0] for pair in smiles_pair_list]
        
        data_dict = {'id': id_list, 'smiles': gt_smiles_list, 'description': description_list}
        # cot_list = ["" for _ in range(len(gt_smiles_list))]
        if self.hparams.cot_mode_multiset in ['simple', 'full', 'formula', 'only_type']:
            multiset_cot_list = map_multiset_cot(gt_smiles_list, mode=self.hparams.cot_mode_multiset)
            data_dict['cot_multiset'] = multiset_cot_list
        
        if self.hparams.cot_mode_ring:
            ring_cot_list = map_ring_cot(gt_smiles_list)
            data_dict['cot_ring'] = ring_cot_list
            
        if self.hparams.cot_mode_fragment:
            fragment_cot_list = map_fragment_cot(split)
            data_dict['cot_fragment'] = fragment_cot_list
            
        if self.hparams.cot_mode_aromatic:
            aromatic_cot_list = map_aromatic_ring_cot(gt_smiles_list)
            data_dict['cot_aromatic'] = aromatic_cot_list
            
        if self.hparams.cot_mode_chain:
            carbon_chain_cot_list = map_carbon_chain_length(gt_smiles_list)
            data_dict['cot_chain'] = carbon_chain_cot_list
            
        if self.hparams.cot_mode_ring_name:
            ring_name_cot_list = map_ring_name_cot(gt_smiles_list)
            data_dict['cot_ring_name'] = ring_name_cot_list
            
        if self.hparams.cot_mode_iupac:
            iupac_cot_list = map_iupac_cot(gt_smiles_list)
            data_dict['cot_iupac'] = iupac_cot_list
        
        if self.hparams.cot_mode_con_ring_name:
            ring_name_cot_list = map_connected_ring_name_cot(gt_smiles_list)
            data_dict['cot_connected_ring_name'] = ring_name_cot_list
        
        dataset = Dataset.from_dict(data_dict)
        
        
        return dataset
    
    def setup_datasets(self, hparams):
        self.train_dataset = self.load_dataset(split='train')
        self.val_dataset = self.load_dataset(split='validation')
        self.test_dataset = self.load_dataset(split='test')
        self.train_dataset_tokenized = self.train_dataset.map(self.preprocess_function, batched=True)
        self.val_dataset_tokenized = self.val_dataset.map(self.preprocess_function, batched=True)
        self.test_dataset_tokenized = self.test_dataset.map(self.preprocess_function, batched=True)
        
    
    def setup_model(self, hparams):
        self.tokenizer = T5Tokenizer.from_pretrained(f"{hparams.model_id}/{hparams.architecture}{hparams.task}", model_max_length=hparams.max_length)
        # TODO: tokenizer training?
        self.pretrained_model = T5ForConditionalGeneration.from_pretrained(f'{hparams.model_id}/{hparams.architecture}{hparams.task}')
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.pretrained_model)
        
    def preprocess_function(self, examples):
        inputs = examples["description"]
        targets = examples['smiles']
        cot_mode = map_cot_mode(self.hparams)

        if self.hparams.architecture.split('-')[0] == 'biot5':
            # convert to selfies
            mols = [Chem.MolFromSmiles(target) for target in targets]
            targets = [selfies.encoder(Chem.MolToSmiles(mol)) for mol in mols]
        
        if cot_mode != "":
            targets = [f" {target}" for target in targets]
        targets = add_cot_to_target(examples, targets, cot_mode)
     
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.hparams.max_length, truncation=True)
        return model_inputs
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='molt5-small', choices=['molt5-small', 'molt5-base', 'molt5-large',
                                                                                        'biot5-base', 'biot5-plus-base', 'biot5-plus-large'])
        parser.add_argument("--cot_mode_multiset", type=str, default='None')
        parser.add_argument("--cot_mode_fragment", action='store_true')
        parser.add_argument("--cot_mode_ring", action='store_true')
        parser.add_argument("--cot_mode_aromatic", action='store_true')
        parser.add_argument("--cot_mode_chain", action='store_true')
        parser.add_argument("--cot_mode_ring_name", action='store_true')
        parser.add_argument("--cot_mode_iupac", action='store_true')
        parser.add_argument("--cot_mode_con_ring_name", action='store_true')
        parser.add_argument("--wandb_mode", type=str, default='disabled')
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--task", type=str, default='', choices=['', '-caption2smiles'])
        parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
        parser.add_argument('--max_length', type=int, default=512)
        parser.add_argument('--test', action='store_false')
        parser.add_argument('--run_id', type=str, default='')
        parser.add_argument('--model_id', type=str, default='laituan245', choices=['laituan245', 'QizhiPei'])

        return parser

class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, test_dataset, hparams):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.hparams = hparams
        self.base_arch = self.hparams.architecture.split('-')[0]
    
    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if len(state.log_history) > 0:
            log = state.log_history[-1]
            if 'loss' in log.keys():
                self._wandb.log({"train/loss": log['loss']})
    
    def log_smiles_results(self, file_name, description_list, gt_smiles, predicted_smiles, decoded_labels, decoded_preds):
        
        with open(f'{file_name}', 'w') as f:
                f.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
                for desc, rt, ot in zip(description_list, gt_smiles, predicted_smiles):
                    f.write(desc + '\t' + rt + '\t' + ot + '\n')
        
        cot_mode = map_cot_mode(self.hparams)
        
        if cot_mode != "":
            columns = ['description', 'gt_smiles', 'predicted_smiles', 'gt_cot', 'predicted_cot']
            # TODO: fix for llama
            num_cot = len(cot_mode.split('-'))-1
            if self.base_arch == 'biot5':
                gt_cots = [".".join(dl.split('.')[:num_cot]) + '.' for dl in decoded_labels]
            else:
                gt_cots = [" ".join(dl.split(' ')[:-1]) for dl in decoded_labels]
            if self.base_arch == 'llama':
                predicted_cots = [dp[:dp.find("The SMILES of the molecule is: ")].split('.')[-2][1:]+'.' if (dp.find("Then, ") > -1) and (dp.find("The SMILES of the molecule is: ")>-1) else " " for dp in decoded_preds]
            elif self.base_arch == 'biot5':
                predicted_cots = [".".join(dp.split('.')[:num_cot]) + '.' for dp in decoded_preds]
            else:   
                predicted_cots = [" ".join(dp.split(' ')[:-1]) for dp in decoded_preds]
            # replacer = {self.tokenizer.eos_token: "", self.tokenizer.bos_token:""}
            if (self.tokenizer.eos_token != None) and (self.tokenizer.bos_token != None):
                predicted_cots = [cot.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.bos_token, "") for cot in predicted_cots]
            result_data = [description_list, gt_smiles, predicted_smiles, gt_cots, predicted_cots]
        else:
            columns = ['description', 'gt_smiles', 'predicted_smiles']
            result_data = [description_list, gt_smiles, predicted_smiles]
        
        if len(columns) > 3:
            cot_mode = map_cot_mode(self.hparams)
            if cot_mode[0] == '-':
                cot_mode = cot_mode[1:]
            # ring_acc, multi_acc, arom_acc = compute_cot_accuracy(gt_cot, predicted_cot, cot_mode=cot_mode)
            cot_acc = compute_cot_accuracy(gt_cots, predicted_cots, cot_mode=cot_mode)
            wandb_log_dict = {}
            cot_modes = cot_mode.split('-')
            for mode, acc in zip(cot_modes, cot_acc):
                if type(acc) == list:
                    wandb_log_dict[f'cot/{mode}_acc'] = sum(acc)/len(acc)
                else:
                    # tuple (tuple of 3 lists)
                    wandb_log_dict[f'cot/{mode}_acc_count'] = sum(acc[0])/len(acc[0])
                    wandb_log_dict[f'cot/{mode}_acc_type'] = sum(acc[1])/len(acc[0])
                    wandb_log_dict[f'cot/{mode}_acc'] = sum(acc[2])/len(acc[0])
            
        
        result_data = list(map(list, zip(*result_data)))
        
        # wandb logging
        table = self._wandb.Table(data=result_data,
                    columns=columns)
        self._wandb.log({f"Prediction": table})
                
        bleu_score, exact_match_score, levenshtein_score, validity_score = mol_translation_metrics.evaluate(file_name)
        validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fingerprint_metrics.evaluate(file_name, 2)
        fcd_metric_score = fcd_metric.evaluate(file_name)
        result = {"BLEU": round(bleu_score, 3), "Exact": round(exact_match_score, 3),
                "Levenshtein": round(levenshtein_score, 3), "MACCS FTS": round(maccs_sims_score, 3),
                "RDK FTS": round(rdk_sims_score, 3), "Morgan FTS": round(morgan_sims_score, 3),
                "FCD Metric": round(fcd_metric_score, 3), "Validity": round(validity_score, 3)
                }
        self._wandb.log(result)
    
    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start evaluation")
            # generate predictions
            predictions = self.trainer.predict(self.test_dataset)
            preds, labels = predictions.predictions, predictions.label_ids
            run_name = map_cot_mode(self.hparams)
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
            preds = preds.astype(int)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

            file_name = f'predictions/ft_cot/{self.hparams.architecture}{self.hparams.task}{run_name}.txt'
            description_list = self.test_dataset['description']

            if run_name == "":
                if self.base_arch == 'biot5':
                    gt_smiles = [selfies.decoder(sf.replace(" ", "")) for sf in decoded_labels]
                    predicted_smiles = [selfies.decoder(sf.replace(" ", "")) for sf in decoded_preds]
                else:
                    gt_smiles = decoded_labels
                    predicted_smiles = decoded_preds
            else:
                if self.base_arch == 'biot5':
                    num_cot = len(run_name.split('-'))-1
                    # selfies to smiles
                    gt_selfies = ["".join(dl.split('.')[num_cot:]).replace(" ", "") for dl in decoded_labels]
                    gt_smiles = [selfies.decoder(sf) for sf in gt_selfies]
                    predicted_selfies =  ["".join(dp.split('.')[num_cot:]).replace(" ", "") for dp in decoded_preds]
                    predicted_smiles = [selfies.decoder(sf) for sf in predicted_selfies]
                else:
                    gt_smiles = [dl.split(' ')[-1] for dl in decoded_labels]
                    predicted_smiles = [dp.split(' ')[-1] for dp in decoded_preds]
            
            self.log_smiles_results(file_name, description_list, gt_smiles, predicted_smiles, decoded_labels, decoded_preds)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FineTuneTranslator.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneTranslator(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)
    run_name = map_cot_mode(hparams)
    
    # for hugging face login
    HfFolder.save_token('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')
    

    if hparams.run_id == '':
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft', mode=hparams.wandb_mode,
               group='ft_cot')
    else:
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft', mode=hparams.wandb_mode,
               group='ft_cot', resume='must', id=hparams.run_id)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"output/{wandb.run.id}",
        eval_strategy="epoch",
        logging_steps=1,
        learning_rate=hparams.learning_rate,
        per_device_train_batch_size=hparams.train_batch_size,
        per_device_eval_batch_size=hparams.eval_batch_size,
        weight_decay=hparams.weight_decay,
        save_total_limit=3,
        num_train_epochs=hparams.epochs,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=True,
        report_to='wandb',
        run_name=f'{hparams.architecture}{run_name}-ft',
        do_train=True,
        generation_max_length=hparams.max_length,
        load_best_model_at_end=True,
        save_strategy='epoch'
    )

    trainer = Seq2SeqTrainer(
        model=model.pretrained_model,
        data_collator=model.data_collator,
        args=training_args,
        train_dataset=model.train_dataset_tokenized,
        eval_dataset=model.test_dataset_tokenized,
        tokenizer=model.tokenizer,
    )
    
    wandb_callback = WandbPredictionProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, hparams=hparams)
    
    wandb.config.update(hparams, allow_val_change=True)
    trainer.add_callback(wandb_callback)
    
    if hparams.run_id == '':
        trainer.train()
    else:
        file_path = sorted([dI for dI in os.listdir(f'output/{hparams.run_id}') if os.path.isdir(os.path.join(f'output/{hparams.run_id}',dI))])[-1]
        # need to check
        # trainer.model._load_optimizer_and_scheduler(f"output/{hparams.run_id}/{file_path}")
        trainer.train(resume_from_checkpoint=f"output/{hparams.run_id}/{file_path}")
    
    wandb.finish()
    