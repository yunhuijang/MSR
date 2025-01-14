import argparse
import numpy as np
import torch
from huggingface_hub.hf_api import HfFolder
import wandb
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["WANDB__SERVICE_WAIT"] = "1000"
from pathlib import Path
from datasets import Dataset, load_dataset
import json
from accelerate import Accelerator

from model.one_stage_generator import FineTuneTranslator, WandbPredictionProgressCallback
from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric
from util_cot import map_cot_mode, map_cot_to_smiles_list, add_cot_to_text
from util import selfies_to_smiles
from tqdm import tqdm


class FineTuneAnswer(FineTuneTranslator):
    def __init__(self, hparams):
        self.run_name = map_cot_mode(hparams)
        super(FineTuneAnswer, self).__init__(hparams)
    
    def load_dataset(self, split):
        if self.dataset_name == 'lm':
            if split == 'train':
                dataset = load_dataset("language-plus-molecules/LPM-24_train", split='train')
            else:
                if self.is_lm_eval:
                    dataset = load_dataset("language-plus-molecules/LPM-24_eval-caption")
                else:
                    dataset = load_dataset(f"language-plus-molecules/LPM-24_train", split='split_valid')
            dataset = dataset.rename_column("molecule", "smiles")
            dataset = dataset.rename_column("caption", "description")  
            # dataset = dataset
            data_dict = {'smiles': dataset['smiles'], 'description': dataset['description']}
            gt_smiles_list = dataset['smiles']
        else:
            if self.base_arch == 'biot5':
                with open(f'ChEBI-20_data/text2mol_{split}.json', 'r') as f:
                    data = json.load(f)
                description_list = [d['input'] for d in data['Instances']]
                gt_selfies_list = [d['output'][0] for d in data['Instances']]
                gt_smiles_list = [selfies_to_smiles(sf[5:-5]) for sf in gt_selfies_list]
                id_list = [d['id'] for d in data['Instances']]
                data_dict = {'id': id_list, 'smiles': gt_selfies_list, 'description': description_list}
            else:
                smiles_list_path = os.path.join('ChEBI-20_data', f'{split}.txt')
                smiles_pair_list = [
                [pair.split()[0], pair.split()[1], " ".join(pair.split()[2:])] for pair in Path(smiles_list_path).read_text(encoding="utf-8").splitlines()
                ][1:]
                description_list = [pair[2] for pair in smiles_pair_list]
                gt_smiles_list = [pair[1] for pair in smiles_pair_list]
                id_list = [pair[0] for pair in smiles_pair_list]
                data_dict = {'id': id_list, 'smiles': gt_smiles_list, 'description': description_list}
            
        if split in ['train', 'validation']:
            data_dict = map_cot_to_smiles_list(gt_smiles_list, self.hparams, data_dict, split)
            
        else:
            if self.hparams.dataset_name == 'lm':
                file_path = 'lm-'
            else:
                file_path = ''
            file_path += f'{self.hparams.architecture}{self.hparams.task}'
            if hasattr(hparams, "select_cot_mode"):
                file_path += self.hparams.cot_mode
            else:
                file_path += self.run_name
            file_name = f'predictions/two_stage_ft_cot/reasoning/{file_path}.txt'
            
            # if hasattr(hparams, "select_cot_mode"):
                # file_name = f'predictions/two_stage_ft_cot/reasoning/{self.hparams.architecture}{self.hparams.task}{self.hparams.cot_mode}.txt'
            # else:
                # file_name = f'predictions/two_stage_ft_cot/reasoning/{self.hparams.architecture}{self.hparams.task}{self.run_name}.txt'
            cot_list = [pair.split('\t')[-1] for pair in Path(file_name).read_text(encoding="utf-8").splitlines()][1:]
            cot_list = [" "+cot for cot in cot_list]
            cot_list_final = cot_list
            cot_mode_split = hparams.cot_mode.split('-')
            cot_mode_select_split = hparams.select_cot_mode.split('-')
            if len(cot_mode_split) != len(cot_mode_select_split):
                if hparams.select_cot_mode == '':
                    cot_list_final = ["" for _ in gt_smiles_list]
                else:
                    cot_indices = [cot_mode_split.index(cot_select) for cot_select in cot_mode_select_split]
                    cot_list_list = [[cot.split('.')[i] for i in cot_indices if i < len(cot.split('.'))] for cot in cot_list]
                    cot_list_final = ['.'.join(cot)+'.' for cot in cot_list_list]
            
            data_dict['cot'] = cot_list_final[:len(gt_smiles_list)]

        dataset = Dataset.from_dict(data_dict)
            
        return dataset
    
    
    def preprocess_function(self, examples, split):
        inputs = examples["description"]
        targets = examples['smiles']
        cots = examples['cot']
        
        inputs = [input_ + cot for input_, cot in zip(inputs, cots)]
        
        if self.hparams.architecture.split('-')[0] == 'biot5':
            # add instruction to input
            task_definition = 'Definition: You are given a molecule description in English. Your job is to generate the molecule SELFIES that fits the description.\n\n'
# 
            inputs = [f'{task_definition}Now complete the following example -\nInput: {inp} \nOutput: ' for inp in inputs]
        
        
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.hparams.max_length, truncation=True)
        return model_inputs
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='multitask-text-and-chemistry-t5-small-standard', choices=['molt5-small', 'molt5-base', 'molt5-large',
                                                                                        'biot5-base', 'biot5-plus-base', 'biot5-plus-large',
                                                                                        'biot5-plus-base-chebi20', 'biot5-base-mol2text', 'biot5-base-text2mol',
                                                                                        'multitask-text-and-chemistry-t5-base-standard', 'multitask-text-and-chemistry-t5-small-standard',
                                                                                        'multitask-text-and-chemistry-t5-base-augm', 'multitask-text-and-chemistry-t5-small-augm'])
        parser.add_argument("--cot_mode", type=str, default='multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral', 
                        help="Choices: func, scaffold, chain, fragment, ring, \
                            multiset_simple/full/formula/type \
                            aromatic, ring_name, con_ring_name, iupac")
        parser.add_argument("--select_cot_mode", type=str, default='aromatic-con_ring_name-func_simple')
        parser.add_argument("--wandb_mode", type=str, default='disabled')
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--train_batch_size", type=int, default=3)
        parser.add_argument("--eval_batch_size", type=int, default=3)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--task", type=str, default='', choices=['', '-caption2smiles'])
        parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
        parser.add_argument('--max_length', type=int, default=512)
        parser.add_argument('--test', action='store_false')
        parser.add_argument('--run_id', type=str, default='')
        parser.add_argument('--model_id', type=str, default='GT4SD', choices=['laituan245', 'QizhiPei', 'GT4SD'])
        # cot correction iteration
        parser.add_argument('--is_iterative', action='store_true')
        parser.add_argument('--num_iter', type=int, default=5)
        parser.add_argument('--warmup_ratio', type=float, default=0)
        parser.add_argument('--lr_scheduler_type', type=str, default='linear')
        parser.add_argument('--max_new_tokens', type=int, default=512)
        parser.add_argument('--generation_mode', action='store_true')
        parser.add_argument('--is_true', action='store_true')
        parser.add_argument('--dataset_name', type=str, default='lm', choices=['molt5', 'lm'])
        parser.add_argument('--is_lm_eval', action='store_true')
        parser.add_argument('--task_name', type=str, default='text2mol')


        return parser


class WandbAnswerProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, hparams):
        super(WandbAnswerProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)

    def on_evaluate(self, args, state, control, **kwargs):
        # super().on_evaluate(args, state, control, **kwargs)
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == self.hparams.epochs):
            print("Start Answer Evaluation")
            # generate predictions
            # generation_index = range(len(self.test_dataset))
            
            
            if self.hparams.is_iterative:
                cot_list = self.test_dataset['cot']
                cot_list_split = [cot.split('.')[:-1] for cot in cot_list]
                gt_smiles, pred_smiles = self.generaate_samples(self.test_dataset, generation_mode=self.hparams.generation_mode, num_iter=self.hparams.num_iter)
                pred_cot_list = [map_cot_to_smiles_list(ps, self.hparams, {}, 'test')['cot'] for ps in pred_smiles]
                pred_cot_list_split = [[c.split('.')[:-1] for c in cot] for cot in pred_cot_list]
                cot_align_list = []
                for cot, pred_cot_l in zip(cot_list_split, pred_cot_list_split):
                    align_list = []
                    for pred_cot in pred_cot_l:
                        align_list.append(sum([c==pc for c, pc in zip(cot, pred_cot)]))
                    cot_align_list.append(align_list)
                
                max_matching_index = np.argmax(cot_align_list, axis=1)
                predicted_smiles = [elem[0] for elem in np.take_along_axis(np.array(pred_smiles), max_matching_index[..., None], axis=1).tolist()]
                if self.hparams.dataset_name == 'lm':
                    file_name = f'predictions/two_stage_ft_cot/answer/lm-{self.hparams.architecture}{self.hparams.task}{run_name}-iter.txt'
                else:
                    file_name = f'predictions/two_stage_ft_cot/answer/{self.hparams.architecture}{self.hparams.task}{run_name}-iter.txt'
            else:
                gt_smiles, predicted_smiles = self.generaate_samples(self.test_dataset, generation_mode=self.hparams.generation_mode)
                if self.hparams.dataset_name == 'lm':
                    file_name = f'predictions/two_stage_ft_cot/answer/lm-{self.hparams.architecture}{self.hparams.task}{run_name}.txt'
                else:
                    file_name = f'predictions/two_stage_ft_cot/answer/{self.hparams.architecture}{self.hparams.task}{run_name}.txt'
                
            description_list = self.test_dataset['description']
            
            with open(f'{file_name}', 'w') as f:
                f.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
                for desc, rt, ot in zip(description_list, gt_smiles, predicted_smiles):
                    f.write(desc + '\t' + rt + '\t' + ot + '\n')
            columns = ['description', 'gt_smiles', 'predicted_smiles']
            result_data = [description_list, gt_smiles, predicted_smiles]
            
            result_data = list(map(list, zip(*result_data)))
            
            # wandb logging
            table = self._wandb.Table(data=result_data,
                        columns=columns)
            self._wandb.log({f"Prediction": table})
            
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
    
    def tokenize_dataset(self, examples):
        description_list = examples["description"]
        cot_list = examples['cot']
        inputs = [desc + cot for desc, cot in zip(description_list, cot_list)]

        model_inputs = self.tokenizer(inputs, max_length=self.hparams.max_length, truncation=True)
        return model_inputs
    
    
    def generaate_samples(self, dataset, generation_mode=False, num_iter=1):
        if generation_mode:
            decoded_preds = []
            input_id_list = self.tokenize_dataset(dataset).input_ids
            for input_ids in tqdm(input_id_list, 'Generation'):
                input_id = torch.tensor(input_ids).unsqueeze(0).to(self.trainer.model.device)
                output_tokens = self.trainer.model.generate(input_id, max_new_tokens=self.hparams.max_new_tokens, num_beams=num_iter, num_return_sequences=num_iter)
                preds = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
                output = [p.strip() for p in preds]
                if self.base_arch == 'biot5':
                    output = [selfies_to_smiles(sf.replace(" ", "")) for sf in output]
                decoded_preds.append(output)
            if self.hparams.is_iterative:
                predicted_smiles = decoded_preds
            else:
                predicted_smiles = [pred[0] for pred in decoded_preds]
            decoded_labels = dataset['smiles']
            if self.base_arch == 'biot5':
                gt_smiles = [selfies_to_smiles(sf.replace(" ", "")) for sf in decoded_labels]
            else:
                gt_smiles = decoded_labels
        else:
            predictions = self.trainer.predict(dataset)
            preds, labels = predictions.predictions, predictions.label_ids
            
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        
            if self.base_arch == 'biot5':
                gt_smiles = [selfies_to_smiles(sf.replace(" ", "")) for sf in decoded_labels]
                predicted_smiles = [selfies_to_smiles(sf.replace(" ", "")) for sf in decoded_preds]
            else:
                gt_smiles = decoded_labels
                predicted_smiles = decoded_preds
            
        return gt_smiles, predicted_smiles
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FineTuneAnswer.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneAnswer(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)
    run_name = map_cot_mode(hparams)
        
    # for hugging face login
    HfFolder.save_token('')
    
    if hparams.architecture.split('-') == 'multitask':
        arch = f'chemt5-{"-".join(hparams.architecture.split("-")[-2:])}'
    else:
        arch = hparams.architecture
    
    if hparams.run_id == '':
        wandb.init(project='mol2text', name=f'{arch}{run_name}-ft-answer', mode=hparams.wandb_mode,
               group='ft_cot_answer')
    else:
        wandb.init(project='mol2text', name=f'{arch}{run_name}-ft-answer', mode=hparams.wandb_mode,
               group='ft_cot_answer', resume='must', id=hparams.run_id)
    
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
        run_name=f'{hparams.architecture}{run_name}-ft-answer',
        do_train=True,
        generation_max_length=hparams.max_length,
        load_best_model_at_end=True,
        save_strategy='epoch',
        warmup_ratio=hparams.warmup_ratio,
        lr_scheduler_type=hparams.lr_scheduler_type
    )

    accelerator = Accelerator()
    
    trainer = accelerator.prepare(Seq2SeqTrainer(
        model=model.pretrained_model,
        data_collator=model.data_collator,
        args=training_args,
        train_dataset=model.train_dataset_tokenized,
        eval_dataset=model.test_dataset_tokenized,
        tokenizer=model.tokenizer,
    ))
    
    wandb_callback = WandbAnswerProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, hparams=hparams)
    
    wandb.config.update(hparams, allow_val_change=True)
    trainer.add_callback(wandb_callback)
    
    if hparams.run_id == '':
        trainer.train()
    else:
        directories = sorted([dI for dI in os.listdir(f'output/{hparams.run_id}') if os.path.isdir(os.path.join(f'output/{hparams.run_id}',dI))])
        last_index = sorted([int(i.split('-')[1]) for i in directories])[-1]
        file_path = f"checkpoint-{last_index}"

        # need to check
        # trainer._load_optimizer_and_scheduler(f"output/{hparams.run_id}/{file_path}")
        trainer.train(resume_from_checkpoint=f"output/{hparams.run_id}/{file_path}")
    
    wandb.finish()