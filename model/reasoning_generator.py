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

from model.one_stage_generator import FineTuneTranslator, WandbPredictionProgressCallback
from analysis import compute_cot_accuracy
from util_cot import map_cot_mode


class FineTuneReasoning(FineTuneTranslator):
    def __init__(self, hparams):
        super(FineTuneReasoning, self).__init__(hparams) 

    
    def preprocess_function(self, examples):
        inputs = examples["description"]
        # targets = examples['smiles']
        
        targets = ["" for _ in range(len(inputs))]
        
        if self.hparams.cot_mode_multiset in ['simple', 'full']:
            targets = [f"{cot_multiset}{target}" for target, cot_multiset in zip(targets, examples['cot_multiset'])]
            
        if self.hparams.cot_mode_ring:
            targets = [f"{cot_ring}{target}" for target, cot_ring in zip(targets, examples['cot_ring'])]
        
        if self.hparams.cot_mode_fragment:
            targets = [f"{cot_fragment}{target}" for target, cot_fragment in zip(targets, examples['cot_fragment'])]

        targets = [target[1:] for target in targets]
        
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.hparams.max_length, truncation=True)
        return model_inputs


class WandbReasoningProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, hparams):
        super(WandbReasoningProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)

    def on_evaluate(self, args, state, control, **kwargs):
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Reasoning Evaluation")
            # generate predictions
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
            
            cot_mode = map_cot_mode(self.hparams.cot_mode_multiset, self.hparams.cot_mode_ring, self.hparams.cot_mode_fragment)
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
    FineTuneReasoning.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneReasoning(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)
    run_name = map_cot_mode(hparams.cot_mode_multiset, hparams.cot_mode_ring, hparams.cot_mode_fragment)
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
        generation_max_length=hparams.max_length
    )

    trainer = Seq2SeqTrainer(
        model=model.pretrained_model,
        data_collator=model.data_collator,
        args=training_args,
        train_dataset=model.train_dataset_tokenized,
        eval_dataset=model.test_dataset_tokenized,
        tokenizer=model.tokenizer,
    )
    
    wandb_callback = WandbReasoningProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, hparams=hparams)
    
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