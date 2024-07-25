import transformers
import torch
from huggingface_hub.hf_api import HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import argparse
import wandb
import os
import numpy as np
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm


from model.one_stage_generator import FineTuneTranslator, WandbPredictionProgressCallback
from util_cot import map_cot_mode
# from util_cot import map_ring_cot, map_multiset_cot, map_fragment_cot

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class FineTuneTranslatorLlama(FineTuneTranslator):
    def __init__(self, hparams):
        super(FineTuneTranslatorLlama, self).__init__(hparams)

    def setup_model(self, hparams):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
)
        
        pretrained_model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                                quantization_config=bnb_config,
                                                                device_map='auto')
        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            )
        pretrained_model.config.pad_token_id = pretrained_model.config.eos_token_id
        
        self.pretrained_model = get_peft_model(pretrained_model, self.peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=hparams.max_length)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
    
    def preprocess_function(self, examples):
        inputs = examples["description"]
        targets = examples['smiles']
        
        cot_list = ["" for _ in range(len(targets))]

        if self.hparams.cot_mode_multiset in ['simple', 'full']:
            cot_list = [f"{cot}{cot_multiset}" for cot, cot_multiset in zip(cot_list, examples['cot_multiset'])]
            
        if self.hparams.cot_mode_ring:
            cot_list = [f"{cot}{cot_ring}" for cot, cot_ring in zip(cot_list, examples['cot_ring'])]
        
        if self.hparams.cot_mode_fragment:
            cot_list = [f"{cot}{cot_fragment}" for cot, cot_fragment in zip(cot_list, examples['cot_fragment'])]
        
        if cot_list[0] == "":
            inputs = [f"{text} The SMILES of the molecule is: {smiles}." for text, smiles in zip(inputs, targets)]
        else:
            cot_list =  [cot[1].lower() + cot[2:] for cot in cot_list]
            inputs = [f"{text} Then, {cot} The SMILES of the molecule is: {smiles}." for text, cot, smiles in zip(inputs, cot_list, targets)]
        examples['text'] = inputs
        
        return examples
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='llama')
        parser.add_argument("--cot_mode_multiset", type=str, default='None')
        parser.add_argument("--cot_mode_fragment", action='store_true')
        parser.add_argument("--cot_mode_ring", action='store_true')
        parser.add_argument("--wandb_mode", type=str, default='disabled')
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--epochs", type=int, default=100)
        # parser.add_argument("--task", type=str, default='', choices=['', '-caption2smiles'])
        parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
        parser.add_argument('--max_length', type=int, default=300)
        parser.add_argument('--test', action='store_false')
        parser.add_argument('--run_id', type=str, default='')

        return parser

class WandbLlamaProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, model, hparams):
        super(WandbLlamaProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)
        self.model = model

    def on_evaluate(self, args, state, control, **kwargs):
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Llama Evaluation")
            # generate predictions
            
            run_name = map_cot_mode(hparams.cot_mode_multiset, hparams.cot_mode_ring, hparams.cot_mode_fragment)
            if run_name == "":
                input_prompt = [f"{des} The SMILES of the molecule is: " for des in self.test_dataset['description']]
            else:
                input_prompt = [f"{des} Then, " for des in self.test_dataset['description']]
            inputs = self.tokenizer(input_prompt, return_tensors='pt',
                                    padding=True).to(self.model.device)
            output = self.model.generate(**inputs, max_length=hparams.max_length)
            
            decoded_preds = self.tokenizer.batch_decode(output)
            
            
            
            file_name = f'predictions/ft_cot_llama/{self.hparams.architecture}{run_name}.txt'

            description_list = self.test_dataset['description']
            gt_smiles = self.test_dataset['smiles']
            predicted_smiles = [dp[dp.index("The SMILES of the molecule is: "):].split('.')[0][len("The SMILES of the molecule is: "):] for dp in decoded_preds]
            predicted_smiles = [smi[1:] if smi[0] == " " else smi for smi in predicted_smiles]
            
            decoded_labels = [text[len(desc)+1:-(len(smi)+len("The SMILES of the molecule is: ")+1)]+ " ." for text, desc, smi in zip(self.test_dataset['text'], description_list, gt_smiles)]
            self.log_smiles_results(file_name, description_list, gt_smiles, predicted_smiles, decoded_labels, decoded_preds)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FineTuneTranslatorLlama.add_args(parser)
    hparams = parser.parse_args()
    model = FineTuneTranslatorLlama(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)
    run_name = map_cot_mode(hparams.cot_mode_multiset, hparams.cot_mode_ring, hparams.cot_mode_fragment)

    # for hugging face login
    HfFolder.save_token('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')
    

    if hparams.run_id == '':
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-llama', mode=hparams.wandb_mode,
               group='ft_cot_llama')
    else:
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-llama', mode=hparams.wandb_mode,
               group='ft_cot_llama', resume='must', id=hparams.run_id)
    
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
        fp16=True,
        push_to_hub=True,
        report_to='wandb',
        run_name=f'{hparams.architecture}{run_name}-ft-llama',
        do_train=True,
        optim="paged_adamw_32bit"
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
    
    wandb_callback = WandbLlamaProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, model=model.pretrained_model, hparams=hparams)
    
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
    