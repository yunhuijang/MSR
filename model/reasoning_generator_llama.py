import argparse
import numpy as np
import torch
from huggingface_hub.hf_api import HfFolder
import wandb
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["WANDB__SERVICE_WAIT"] = "300"

from model.one_stage_generator_llama import FineTuneTranslatorLlama, WandbLlamaProgressCallback
from analysis import compute_cot_accuracy
from util_cot import map_cot_mode, add_cot_to_target


class FineTuneReasoningLlama(FineTuneTranslatorLlama):
    def __init__(self, hparams):
        super(FineTuneReasoningLlama, self).__init__(hparams) 

    
    def preprocess_function(self, examples):
        inputs = examples["description"]
        # targets = examples['smiles']
        
        targets = ["" for _ in range(len(inputs))]
        cot_mode = map_cot_mode(self.hparams)
        targets = add_cot_to_target(examples, targets, cot_mode)

        targets = [" Then, " + target[1].lower() + target[2:] for target in targets]
        
        examples['text'] = [input + target for input, target in zip(inputs, targets)]
        
        # model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.hparams.max_length, truncation=True)
        return examples

    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='llama')
        parser.add_argument("--cot_mode_multiset", type=str, default='None')
        parser.add_argument("--cot_mode_fragment", action='store_true')
        parser.add_argument("--cot_mode_ring", action='store_true')
        parser.add_argument("--wandb_mode", type=str, default='disabled')
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_batch_size", type=int, default=2)
        parser.add_argument("--eval_batch_size", type=int, default=4)
        parser.add_argument("--gen_batch_size", type=int, default=32)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--epochs", type=int, default=30)
        # parser.add_argument("--task", type=str, default='', choices=['', '-caption2smiles'])
        parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
        parser.add_argument('--max_length', type=int, default=512)
        parser.add_argument('--test', action='store_false')
        parser.add_argument('--run_id', type=str, default='')

        return parser


class WandbReasoningLlamaProgressCallback(WandbLlamaProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, model, hparams):
        super(WandbReasoningLlamaProgressCallback, self).__init__(trainer, tokenizer, test_dataset, model, hparams)

    def on_evaluate(self, args, state, control, **kwargs):
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Reasoning Evaluation")
            # generate predictions
            

            input_prompt = [f"{des} Then, " for des in self.test_dataset['description']]
            
            
            total_num_samples = 0
            decoded_preds = []
            while(len(decoded_preds)<len(input_prompt)):
                cur_num_samples = min(len(input_prompt) - total_num_samples, self.hparams.gen_batch_size)
                inputs = self.tokenizer(input_prompt[total_num_samples:total_num_samples+cur_num_samples], return_tensors='pt',
                                        padding=True).to(self.model.device)
                output = self.model.generate(**inputs, max_length=hparams.max_length)
                total_num_samples += cur_num_samples
                decoded_batch = self.tokenizer.batch_decode(output)
                
                decoded_preds.extend(decoded_batch)
            
                print(f'Sampling: {total_num_samples}')
                
            
            file_name = f'predictions/two_stage_ft_cot/reasoning/{self.hparams.architecture}{run_name}.txt'
            description_list = self.test_dataset['description']
            
            
            targets = ["" for _ in range(len(decoded_preds))]
            cot_mode = map_cot_mode(self.hparams)
            targets = add_cot_to_target(self.test_dataset, targets, cot_mode)
            
            gt_cot = targets
            targets = [target[1].lower() + target[2:] for target in targets]
            predicted_cot = [dp[dp.find("Then, it"):].split('.')[0][len("Then, "):] if dp.find("Then, it") > -1 else " " for dp in decoded_preds]
            
            with open(f'{file_name}', 'w') as f:
                f.write('description' + '\t' + 'ground truth cot' + '\t' + 'output cot' + '\n')
                for desc, rt, ot in zip(description_list, gt_cot, predicted_cot):
                    f.write(desc + '\t' + rt + '\t' + ot + '\n')
            
            columns = ['description', 'gt_cot', 'predicted_cot']
            result_data = [description_list, gt_cot, predicted_cot]
            
            result_data = list(map(list, zip(*result_data)))
            
            # wandb logging
            table = self._wandb.Table(data=result_data,
                        columns=columns)
            self._wandb.log({f"Prediction": table})
            
            # log accuracy
            
            cot_mode = map_cot_mode(self.hparams)
            if cot_mode[0] == '-':
                cot_mode = cot_mode[1:]
            ring_acc, multi_acc = compute_cot_accuracy(gt_cot, predicted_cot, cot_mode=cot_mode)
            
            wandb_log_dict = {}
            if len(ring_acc[0]) > 0:
                wandb_log_dict['cot/ring_acc_count'] = sum(ring_acc[0])/len(ring_acc[0])
                wandb_log_dict['cot/ring_acc_type'] = sum(ring_acc[1])/len(ring_acc[0])
                wandb_log_dict['cot/ring_acc'] = sum(ring_acc[2])/len(ring_acc[0])
            
            if len(multi_acc[0]) > 0:
                wandb_log_dict['cot/multi_acc_count'] = sum(multi_acc[0])/len(multi_acc[0])
                wandb_log_dict['cot/multi_acc_type'] = sum(multi_acc[1])/len(multi_acc[0])
                wandb_log_dict['cot/multi_acc'] = sum(multi_acc[2])/len(multi_acc[0])
            
            self._wandb.log(wandb_log_dict)
            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FineTuneReasoningLlama.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneReasoningLlama(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)
    run_name = map_cot_mode(hparams)
    HfFolder.save_token('')
    
    if hparams.run_id == '':
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-reasoning', mode=hparams.wandb_mode,
               group='ft_cot_reasoning')
    else:
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-reasoning', mode=hparams.wandb_mode,
               group='ft_cot_reasoning', resume='must', id=hparams.run_id)
    
    training_args = TrainingArguments(
            output_dir=f"output/{wandb.run.id}",
            eval_strategy="epoch",
            logging_steps=1,
            learning_rate=hparams.learning_rate,
            per_device_train_batch_size=hparams.train_batch_size,
            per_device_eval_batch_size=hparams.eval_batch_size,
            weight_decay=hparams.weight_decay,
            save_total_limit=3,
            num_train_epochs=hparams.epochs,
            fp16=False,
            push_to_hub=True,
            report_to='wandb',
            run_name=f'{hparams.architecture}{run_name}-ft-llama',
            do_train=True,
            optim="paged_adamw_32bit",
            load_best_model_at_end=True,
            save_strategy='epoch'
        )
        
    trainer = SFTTrainer(
            model=model.pretrained_model,
            # data_collator=model.data_collator,
            args=training_args,
            train_dataset=model.train_dataset_tokenized,
            eval_dataset=model.test_dataset_tokenized,
            tokenizer=model.tokenizer,
            peft_config=model.peft_config,
            max_seq_length=1024,
            dataset_text_field='text',
            packing=False
        )
        
    wandb_callback = WandbReasoningLlamaProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, model=model.pretrained_model, hparams=hparams)
    
    wandb.config.update(hparams, allow_val_change=True)
    trainer.add_callback(wandb_callback)
    
    if hparams.run_id == '':
        trainer.train()
    else:
        file_path = sorted([dI for dI in os.listdir(f'output/{hparams.run_id}') if os.path.isdir(os.path.join(f'output/{hparams.run_id}',dI))])[-1]
        # need to check
        # trainer._load_optimizer_and_scheduler(f"output/{hparams.run_id}/{file_path}")
        trainer.train(resume_from_checkpoint=f"output/{hparams.run_id}/{file_path}")
    
    wandb.finish()