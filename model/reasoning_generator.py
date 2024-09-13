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
from accelerate import Accelerator
from tqdm import tqdm

from model.one_stage_generator import FineTuneTranslator, WandbPredictionProgressCallback
from analysis import compute_cot_accuracy
from util_cot import map_cot_mode, add_cot_to_target


class FineTuneReasoning(FineTuneTranslator):
    def __init__(self, hparams):
        super(FineTuneReasoning, self).__init__(hparams) 

    
    def preprocess_function(self, examples, split):
        inputs = examples["description"]
        # targets = examples['smiles']
        
        targets = ["" for _ in range(len(inputs))]
        
        cot_mode = map_cot_mode(self.hparams)
        targets = add_cot_to_target(examples, targets, cot_mode)

        targets = [target[1:] for target in targets]
        
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.hparams.max_length, truncation=True)
        return model_inputs

    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='molt5-base')
        parser.add_argument("--cot_mode", type=str, default='multiset_formula-func_smiles-chain-aromatic-con_ring_name', 
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
        parser.add_argument('--test', action='store_false')
        parser.add_argument('--run_id', type=str, default='')
        parser.add_argument('--model_id', type=str, default='laituan245', choices=['laituan245', 'QizhiPei'])
        parser.add_argument('--warmup_ratio', type=float, default=0)
        parser.add_argument('--lr_scheduler_type', type=str, default='linear')
        parser.add_argument('--max_new_tokens', type=int, default=512)
        parser.add_argument('--generation_mode', action='store_true')

        return parser
    
    
class WandbReasoningProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, hparams):
        super(WandbReasoningProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)

    def tokenize_dataset(self, examples):
        inputs = examples["description"]
        # targets = examples['smiles']
        
        targets = ["" for _ in range(len(inputs))]
        
        cot_mode = map_cot_mode(self.hparams)
        targets = add_cot_to_target(examples, targets, cot_mode)

        targets = [target[1:] for target in targets]
        
        model_inputs = self.tokenizer(inputs, max_length=self.hparams.max_length, truncation=True)
        return model_inputs
    
    
    def on_evaluate(self, args, state, control, **kwargs):
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Reasoning Evaluation")
            # generate predictions
            
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
                decoded_labels = self.test_dataset['cot']
            else:
                predictions = self.trainer.predict(self.test_dataset)
                preds, labels = predictions.predictions, predictions.label_ids
                
                if isinstance(preds, tuple):
                    preds = preds[0]
                preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

            file_name = f'predictions/two_stage_ft_cot/reasoning/{self.hparams.architecture}{self.hparams.task}{run_name}.txt'
            description_list = self.test_dataset['description']
            gt_cot = decoded_labels
            predicted_cot = decoded_preds
            
            with open(f'{file_name}', 'w') as f:
                f.write('description' + '\t' + 'ground truth cot' + '\t' + 'output cot' + '\n')
                for desc, rt, ot in zip(description_list, gt_cot, predicted_cot):
                    f.write(desc + '\t' + rt + '\t' + ot + '\n')
            
            columns = ['description', 'gt_cot', 'predicted_cot']
            result_data = [description_list, decoded_labels, decoded_preds]
            
            result_data = list(map(list, zip(*result_data)))
            
            # wandb logging
            table = self._wandb.Table(data=result_data,
                        columns=columns)
            self._wandb.log({f"Prediction": table})
            
            # log accuracy
            
            cot_mode = map_cot_mode(self.hparams)
            if cot_mode[0] == '-':
                cot_mode = cot_mode[1:]
            # ring_acc, multi_acc, arom_acc = compute_cot_accuracy(gt_cot, predicted_cot, cot_mode=cot_mode)
            cot_acc = compute_cot_accuracy(gt_cot, predicted_cot, cot_mode=cot_mode)
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
            
            self._wandb.log(wandb_log_dict)
            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FineTuneReasoning.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneReasoning(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)
    run_name = map_cot_mode(hparams)
    HfFolder.save_token('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')
    
    if hparams.run_id == '':
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-reasoning', mode=hparams.wandb_mode,
               group='ft_cot_reasoning')
    else:
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-reasoning', mode=hparams.wandb_mode,
               group='ft_cot_reasoning', resume='must', id=hparams.run_id)
    
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
        run_name=f'{hparams.architecture}{run_name}-ft-reasoning',
        do_train=True,
        generation_max_length=hparams.max_length,
        save_strategy='epoch',
        load_best_model_at_end=True,
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
    
    wandb_callback = WandbReasoningProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, hparams=hparams)
    
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