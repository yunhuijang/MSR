import argparse
import numpy as np
import torch
from huggingface_hub.hf_api import HfFolder
import wandb
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["WANDB__SERVICE_WAIT"] = "300"
from pathlib import Path
from datasets import Dataset
import json

from model.one_stage_generator import FineTuneTranslator, WandbPredictionProgressCallback
from util_cot import map_cot_mode
from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric
from util_cot import map_cot_mode, map_cot_to_smiles_list
from util import selfies_to_smiles

class FineTuneAnswer(FineTuneTranslator):
    def __init__(self, hparams):
        self.run_name = map_cot_mode(hparams)
        super(FineTuneAnswer, self).__init__(hparams)
    
    def load_dataset(self, split):
        
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
            # if self.hparams.test:
            #     smiles_pair_list = smiles_pair_list[:20]
            description_list = [pair[2] for pair in smiles_pair_list]
            gt_smiles_list = [pair[1] for pair in smiles_pair_list]
            id_list = [pair[0] for pair in smiles_pair_list]
            data_dict = {'id': id_list, 'smiles': gt_smiles_list, 'description': description_list}
            
        if split in ['train', 'validation']:
            data_dict = map_cot_to_smiles_list(gt_smiles_list, self.hparams, data_dict, split)
            
        else:
            file_name = f'predictions/two_stage_ft_cot/reasoning/{self.hparams.architecture}{self.hparams.task}{self.run_name}.txt'
            cot_list = [pair.split('\t')[-1] for pair in Path(file_name).read_text(encoding="utf-8").splitlines()][1:]
            cot_list = [" "+cot for cot in cot_list]
            data_dict['cot'] = cot_list[:len(gt_smiles_list)]

            
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
        parser.add_argument("--architecture", type=str, default='molt5-base', choices=['molt5-small', 'molt5-base', 'molt5-large',
                                                                                        'biot5-base', 'biot5-plus-base', 'biot5-plus-large'])
        parser.add_argument("--cot_mode_multiset", type=str, default='None')
        parser.add_argument("--cot_mode_fragment", action='store_true')
        parser.add_argument("--cot_mode_ring", action='store_true')
        parser.add_argument("--cot_mode_aromatic", action='store_true')
        parser.add_argument("--cot_mode_chain", action='store_true')
        parser.add_argument("--cot_mode_ring_name", action='store_true')
        parser.add_argument("--cot_mode_iupac", action='store_true')
        parser.add_argument("--cot_mode_con_ring_name", action='store_true')
        parser.add_argument("--cot_mode_scaffold", action='store_true')
        parser.add_argument("--cot_mode_functional_group", action='store_true')
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
        # cot correction iteration
        parser.add_argument('--is_iterative', action='store_true')
        parser.add_argument('--num_iter', type=int, default=5)
        parser.add_argument('--warmup_ratio', type=float, default=0)
        parser.add_argument('--lr_scheduler_type', type=str, default='linear')


        return parser


class WandbAnswerProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, hparams):
        super(WandbAnswerProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)

    def on_evaluate(self, args, state, control, **kwargs):
        # super().on_evaluate(args, state, control, **kwargs)
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Answer Evaluation")
            # generate predictions
            generation_index = range(len(self.test_dataset))
            gt_smiles, predicted_smiles = self.generaate_samples(generation_index)
            
            if self.hparams.is_iterative and state.epoch > 10:
                num_iter = 0
                cot_list = self.test_dataset['cot']
                while((num_iter < self.hparams.num_iter) or (len(generation_index) == 0)):
                    data_dict = {'smiles': predicted_smiles}
                    data_dict = map_cot_to_smiles_list(predicted_smiles, self.hparams, data_dict, 'test')
                    generation_index_bool = [pred_info != true_info for pred_info, true_info in zip(data_dict['cot'], cot_list)]
                    generation_index = [i for i, x in enumerate(generation_index_bool) if x]
                    _, new_predicted_smiles = self.generaate_samples(generation_index)
                    final_predicted_smiles = []
                    for index, gi_bool in enumerate(generation_index_bool):
                        # matching CoT -> keep the predicted smiles
                        if not gi_bool:
                            final_predicted_smiles.append(predicted_smiles[index])
                        # not matching CoT -> replace with new predicted smiles
                        else:
                            final_predicted_smiles.append(new_predicted_smiles.pop(0))
                    predicted_smiles = final_predicted_smiles
                    
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
            
    def generaate_samples(self, generation_index):
        dataset = Dataset.from_dict(self.test_dataset[generation_index])
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
    HfFolder.save_token('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')
    
    if hparams.run_id == '':
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-answer', mode=hparams.wandb_mode,
               group='ft_cot_answer')
    else:
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-answer', mode=hparams.wandb_mode,
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
        warmup_ratio=0.02
    )

    trainer = Seq2SeqTrainer(
        model=model.pretrained_model,
        data_collator=model.data_collator,
        args=training_args,
        train_dataset=model.train_dataset_tokenized,
        eval_dataset=model.test_dataset_tokenized,
        tokenizer=model.tokenizer,
    )
    
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