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


from model.one_stage_generator_llama import FineTuneTranslatorLlama, WandbPredictionProgressCallback
from util_cot import map_cot_mode
from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric


class FineTuneAnswerLlama(FineTuneTranslatorLlama):
    def __init__(self, hparams):
        self.run_name = map_cot_mode(hparams)
        super(FineTuneAnswerLlama, self).__init__(hparams)
        
    
    def preprocess_function(self, examples):
        inputs = examples["description"]
        targets = examples['smiles']
        
        file_name = f'predictions/two_stage_ft_cot/reasoning/{self.hparams.architecture}{self.run_name}.txt'
        cots = [pair.split('\t')[-1] for pair in Path(file_name).read_text(encoding="utf-8").splitlines()][1:]
        inputs = [input_ + ' ' + cot for input_, cot in zip(inputs, cots)]
        
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.hparams.max_length, truncation=True)
        return model_inputs


class WandbAnswerProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, hparams):
        super(WandbAnswerProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)

    def on_evaluate(self, args, state, control, **kwargs):
        # super().on_evaluate(args, state, control, **kwargs)
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Answer Evaluation")
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

            file_name = f'predictions/two_stage_ft_cot/answer/{self.hparams.architecture}{self.hparams.task}{run_name}.txt'
            description_list = self.test_dataset['description']
            
            gt_smiles = decoded_labels
            predicted_smiles = decoded_preds
            
            with open(f'{file_name}', 'w') as f:
                f.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
                for desc, rt, ot in zip(description_list, gt_smiles, predicted_smiles):
                    f.write(desc + '\t' + rt + '\t' + ot + '\n')
            
            columns = ['description', 'gt_smiles', 'predicted_smiles', 'gt_cot', 'predicted_cot']
            gt_cots = [" ".join(dl.split(' ')[:-1]) for dl in decoded_labels]
            predicted_cots = [" ".join(dp.split(' ')[:-1]) for dp in decoded_preds]
            result_data = [description_list, gt_smiles, predicted_smiles, gt_cots, predicted_cots]
            
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
            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FineTuneAnswerLlama.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneAnswerLlama(hparams)
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
        fp16=True,
        push_to_hub=True,
        report_to='wandb',
        run_name=f'{hparams.architecture}{run_name}-ft-answer',
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
    
    wandb_callback = WandbAnswerProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, hparams=hparams)
    
    wandb.config.update(hparams, allow_val_change=True)
    trainer.add_callback(wandb_callback)
    
    if hparams.run_id == '':
        trainer.train()
    else:
        file_path = sorted([dI for dI in os.listdir(f'output/{hparams.run_id}') if os.path.isdir(os.path.join(f'output/{hparams.run_id}',dI))])[-1]
        # need to check
        trainer._load_optimizer_and_scheduler(f"output/{hparams.run_id}/{file_path}")
        trainer.train(resume_from_checkpoint=f"output/{hparams.run_id}/{file_path}")
    
    wandb.finish()