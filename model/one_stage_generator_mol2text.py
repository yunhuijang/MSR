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

import wandb
import pytorch_lightning as pl
from huggingface_hub.hf_api import HfFolder
import torch
from accelerate import Accelerator


from evaluation import text_translation_metrics
from util_cot import map_cot_mode, add_cot_to_target, add_cot_to_text
from util import selfies_to_smiles
from model.one_stage_generator import FineTuneTranslator, WandbPredictionProgressCallback


class FineTuneTranslatorMol2Text(FineTuneTranslator):
    def __init__(self, hparams):
        super(FineTuneTranslatorMol2Text, self).__init__(hparams)
    
    def preprocess_function(self, examples, split):
        inputs = examples["smiles"]
        cot_mode = map_cot_mode(self.hparams)
        targets = examples['description']
        if self.base_arch == 'biot5':
            # add instruction to input
            task_definition = 'Definition: You are given a molecule SELFIES and its structural information. Your job is to generate the molecule description in English that fits the molecule SELFIES.\n\n'

            inputs = [f'{task_definition}Now complete the following example -\nInput: {inp}' for inp in inputs]
            
            # targets = [f"\nOutput: {target}" for target in targets][:len(inputs)]
        
        elif self.base_arch == 'multitask':
            inputs = [f"Caption the following SMILES: {smiles}." for smiles in inputs]
            

        
        
        if cot_mode != "":
            inputs = [f" {smiles}" for smiles in inputs]
        # No need for learning CoT
        if self.hparams.force:
            if self.hparams.architecture.split('-')[0] == 'biot5':
                inputs = [inp + "\nOutput: " for inp in inputs]
        inputs = add_cot_to_text(examples, inputs, 'backward')
        inputs = [inp.strip() for inp in inputs]
        if not self.hparams.force:
            if self.hparams.architecture.split('-')[0] == 'biot5':
                inputs = [inp + "\nOutput: " for inp in inputs]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.hparams.max_length, truncation=True)
        return model_inputs
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='molt5-base', choices=['molt5-small', 'molt5-base', 'molt5-large',
                                                                                        'biot5-base', 'biot5-plus-base', 'biot5-plus-large',
                                                                                        'biot5-plus-base-chebi20', 'biot5-base-mol2text', 'biot5-base-text2mol',
                                                                                        'multitask-text-and-chemistry-t5-base-standard', 'multitask-text-and-chemistry-t5-small-standard',
                                                                                        'multitask-text-and-chemistry-t5-base-augm', 'multitask-text-and-chemistry-t5-small-augm'
                                                                                        ])
# multiset_formula-func_simple-chain-aromatic-con_ring_name
        parser.add_argument("--cot_mode", type=str, default='', 
                        help="Choices: func, scaffold, chain, fragment, ring, \
                            multiset_simple/full/formula/type \
                            aromatic, ring_name, con_ring_name, iupac")
        parser.add_argument("--wandb_mode", type=str, default='disabled')
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--task", type=str, default='', choices=['', '-caption2smiles'])
        parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
        parser.add_argument('--max_length', type=int, default=512)
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--run_id', type=str, default='')
        parser.add_argument('--model_id', type=str, default='laituan245', choices=['laituan245', 'QizhiPei', 'GT4SD'])
        parser.add_argument('--warmup_ratio', type=float, default=0)
        parser.add_argument('--lr_scheduler_type', type=str, default='linear')
        parser.add_argument('--max_new_tokens', type=int, default=512)
        parser.add_argument('--generation_mode', action='store_true')
        parser.add_argument('--force', action='store_true')

        return parser

class WandbPredictionProgressCallbackMol2Text(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, hparams):
        super().__init__(trainer, tokenizer, test_dataset, hparams)
        self.model = self.trainer.model
        decoded_labels = []
        for gt in self.test_dataset['description']:
            decoded_label = self.tokenizer.batch_decode(torch.tensor(self.tokenizer(gt).input_ids).unsqueeze(0), skip_special_tokens=True)[0]
            decoded_labels.append(decoded_label)
        self.decoded_labels = decoded_labels
    
    def log_description_results(self, file_name, smiles_list, gt_description, predicted_description):
        
        with open(f'{file_name}', 'w') as f:
                f.write('SMILES' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
                for desc, rt, ot in zip(smiles_list, gt_description, predicted_description):
                    f.write(desc + '\t' + rt + '\t' + ot + '\n')
        
        columns = ['smiles', 'gt_description', 'predicted_description']
        result_data = [smiles_list, gt_description, predicted_description]
        result_data = list(map(list, zip(*result_data)))
        
        # wandb logging
        table = self._wandb.Table(data=result_data,
                    columns=columns)
        self._wandb.log({f"Prediction": table})
                
        bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                text_translation_metrics.evaluate(
                    'allenai/scibert_scivocab_uncased', file_name, 512
                )
        result = {"BLEU2": round(bleu2, 3), "BLEU4": round(bleu4, 3),
                "ROUGE1": round(rouge_1, 3), "ROUGE2": round(rouge_2, 3),
                "ROUGEL": round(rouge_l, 3), "METEOR": round(meteor_score, 3)
                }
        self._wandb.log(result)
        
        
    def tokenize_dataset(self, examples):
        inputs = examples["smiles"]
        cot_mode = map_cot_mode(self.hparams)
        if self.base_arch == 'biot5':
            # add instruction to input
            task_definition = 'Definition: You are given a molecule SELFIES and its structural information. Your job is to generate the molecule description in English that fits the molecule SELFIES.\n\n'

            inputs = [f'{task_definition}Now complete the following example -\nInput: {inp}' for inp in inputs]
            
        if cot_mode != "":
            inputs = [f" {smiles}" for smiles in inputs]
        inputs = add_cot_to_text(examples, inputs, 'backward')
        inputs = [inp.strip() for inp in inputs]
        if self.base_arch == 'biot5':
            inputs = [inp + "\nOutput: " for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.hparams.max_length, truncation=True)
        return model_inputs
    
    
    def on_evaluate(self, args, state, control, **kwargs):
        # super().on_evaluate(args, state, control, **kwargs)
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start evaluation")
            # # generate predictions
            # inputs = self.test_dataset
            # output = self.model.generate(self.test_dataset_tokenized['input_ids'], max_length=self.hparams.max_length)
            # print('hi')
            run_name = map_cot_mode(self.hparams)
            
            if self.hparams.generation_mode:
                decoded_preds = []
                input_id_list = self.tokenize_dataset(self.test_dataset).input_ids
                for input_ids in tqdm(input_id_list, 'Generation'):
                    input_id = torch.tensor(input_ids).unsqueeze(0).to(self.trainer.model.device)
                    output_tokens = self.trainer.model.generate(input_id, max_new_tokens=self.hparams.max_new_tokens)
                    preds = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
                    if isinstance(preds, list):
                        output = preds[0]
                    decoded_preds.append(output)

                
                decoded_labels = self.decoded_labels
            else:
                predictions = self.trainer.predict(self.test_dataset)
                preds, labels = predictions.predictions, predictions.label_ids
                if isinstance(preds, tuple):
                    preds = preds[0]
                preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
                preds = preds.astype(int)
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
                
            gt_description = decoded_labels
            predicted_description = decoded_preds
            
                
            if hparams.force:
                file_name = f'predictions/ft_cot_mol2text/{self.hparams.architecture}{self.hparams.task}{run_name}-force.txt'
            else:
                file_name = f'predictions/ft_cot_mol2text/{self.hparams.architecture}{self.hparams.task}{run_name}.txt'
            smiles_list = self.test_dataset['smiles']

            self.log_description_results(file_name, smiles_list, gt_description, predicted_description)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FineTuneTranslatorMol2Text.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneTranslatorMol2Text(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)
    run_name = map_cot_mode(hparams)
    
    # for hugging face login
    HfFolder.save_token('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')
    

    if hparams.run_id == '':
        wandb.init(project='text2mol', name=f'{hparams.architecture}{run_name}-ft-m2t', mode=hparams.wandb_mode,
               group='ft_cot')
    else:
        if hparams.architecture.split('-') == 'multitask':
            wandb.init(project='text2mol', name=f'chemt5-{"-".join(hparams.architecture.split("-")[-2:])}-{run_name}-ft-m2t', mode=hparams.wandb_mode,
               group='ft_cot', resume='must', id=hparams.run_id)
        else:
            wandb.init(project='text2mol', name=f'{hparams.architecture}{run_name}-ft-m2t', mode=hparams.wandb_mode,
                group='ft_cot', resume='must', id=hparams.run_id)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"output/{wandb.run.id}",
        eval_strategy="epoch",
        logging_steps=1,
        learning_rate=hparams.learning_rate,
        per_device_train_batch_size=hparams.train_batch_size,
        per_device_eval_batch_size=hparams.eval_batch_size,
        weight_decay=hparams.weight_decay,
        save_total_limit=2,
        num_train_epochs=hparams.epochs,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=True,
        report_to='wandb',
        run_name=f'{hparams.architecture}{run_name}-ft',
        do_train=True,
        generation_max_length=hparams.max_length,
        load_best_model_at_end=True,
        save_strategy='epoch',
        warmup_ratio=hparams.warmup_ratio,
        lr_scheduler_type=hparams.lr_scheduler_type
    )

    accelerater = Accelerator()
    
    
    trainer = accelerater.prepare(Seq2SeqTrainer(
        model=model.pretrained_model,
        data_collator=model.data_collator,
        args=training_args,
        train_dataset=model.train_dataset_tokenized,
        eval_dataset=model.test_dataset_tokenized,
        tokenizer=model.tokenizer,
    ))
    
    wandb_callback = WandbPredictionProgressCallbackMol2Text(trainer, model.tokenizer, model.test_dataset_tokenized, hparams=hparams)
    
    wandb.config.update(hparams, allow_val_change=True)
    trainer.add_callback(wandb_callback)
    
    if hparams.run_id == '':
        trainer.train()
    else:
        directories = sorted([dI for dI in os.listdir(f'output/{hparams.run_id}') if os.path.isdir(os.path.join(f'output/{hparams.run_id}',dI))])
        last_index = sorted([int(i.split('-')[1]) for i in directories])[-1]
        file_path = f"checkpoint-{last_index}"        # need to check
        # trainer.model._load_optimizer_and_scheduler(f"output/{hparams.run_id}/{file_path}")
        trainer.train(resume_from_checkpoint=f"output/{hparams.run_id}/{file_path}")
    
    wandb.finish()
    