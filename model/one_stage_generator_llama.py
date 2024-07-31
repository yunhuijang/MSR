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
from util_cot import map_cot_mode, add_cot_to_target
# from util_cot import map_ring_cot, map_multiset_cot, map_fragment_cot

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class FineTuneTranslatorLlama(FineTuneTranslator):
    def __init__(self, hparams):
        super(FineTuneTranslatorLlama, self).__init__(hparams)

    def fix_untrained_tokens(self, model, eps = 1e-16):
        """
        Llama-3 for eg has untrained vectors in the base model.
        These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
        We reset them to the mean of the rest of the tokens
        """
        embedding_matrix = model.get_input_embeddings().weight.data
        lm_head_matrix   = model.get_output_embeddings().weight.data

        # Get untrained tokens
        indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
        where_untrained = torch.where(indicator_untrained)[0]
        n_untrained = where_untrained.shape[0]
        n_trained = embedding_matrix.shape[0] - n_untrained
        if n_untrained != 0:
            print(
                f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
                "We shall set them to the mean of the other trained tokens."
            )
        pass

        # First set untrained to all 0s - sometimes it's not! 1e-23 for bfloat16
        embedding_matrix[where_untrained] = 0
        lm_head_matrix  [where_untrained] = 0

        # Find sum
        sum_embedding  = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
        sum_lm_head    = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

        # Find correct average by dividing by sum of trained tokens
        mean_embedding = (sum_embedding / n_trained).to(embedding_matrix.dtype)
        mean_lm_head   = (sum_lm_head   / n_trained).to(lm_head_matrix  .dtype)

        # Set them to the mean
        embedding_matrix[where_untrained] = mean_embedding
        lm_head_matrix  [where_untrained] = mean_lm_head

        return mean_embedding, mean_lm_head
    
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
        # pretrained_model.config.pad_token_id = pretrained_model.config.unk_token_id
        
        self.pretrained_model = get_peft_model(pretrained_model, self.peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=hparams.max_length)
        if tokenizer.pad_token is None:
            # tokenizer.pad_token = "<|reserved_special_token_0|>"
            tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_0|>"})
            self.pretrained_model.config.pad_token_id = tokenizer.pad_token_id            
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = "left"
        
        ion_tokens = ['[14C]', '[AsH3]', '[Se+]', '[SiH4]', '[AsH]', '[Mo+4]', '[Ru]', '[Be]', '[1H+]', '[S-2]', '[nH+]',
                      '[Ti]', '[Pr+3]', '[Ir-2]', '[131I]', '[S@]', '[SH+]', '[Co+2]', '[S+]', '[C@@H]', '[125Te]', '[C]',
                      '[18OH2]', '[Zn]', '[S-]', '[33PH3]', '[P@@]', '[Fe]', '[1HH]', '[Ni+2]', '[Pt+2]', '[H+]', '[C@]', 
                      '[Au+]', '[Cr]', '[18F]', '[Hg+2]', '[Se]', '[Fe+4]', '[197Hg]', '[I-]', '[Tl+]', '[3HH]', '[210Po]',
                      '[NH2-]', '[NH4+]', '[o+]', '[N@+]', '[I+]', '[O]', '[N@@+]', '[Cr+3]', '[14CH3]', '[Cu]', '[15NH2]',
                      '[Cl-]', '[N]', '[Pd-2]', '[77Se]', '[P+]', '[P@@H]', '[S@@+]', '[Sb-]', '[Hg]', '[Be+2]', '[Pb]', 
                      '[s+]', '[SnH]', '[13C@@H]', '[205Tl]', '[3H]', '[Na+]', '[V]', '[Sb]', '[87Rb]', '[35SH2]', '[Os]', 
                      '[C-]', '[SH-]', '[Rh-3]', '[Ni]', '[Sn]', '[129Xe]', '[SeH2]', '[As]', '[OH-]', '[Zr]', '[13CH]', 
                      '[23Na]', '[Li+]', '[127IH]', '[10B]', '[115Sn]', '[Se-2]', '[18FH]', '[Mn]', '[CH]', '[Fe+3]', 
                      '[Pd]', '[Mo+2]', '[Se-]', '[Cd]', '[51Cr]', '[F-]', '[Ir+3]', '[OH3+]', '[NH-]', '[45Sc]', 
                      '[139La]', '[Ca+2]', '[CH+]', '[Fe+2]', '[CH3-]', '[11CH3]', '[C@@]', '[119Sn]', '[S@@]', '[Co]', 
                      '[9Be]', '[Br-]', '[151Eu]', '[W]', '[N-]', '[95Mo]', '[Rb+]', '[Ir]', '[3He]', '[4He]', '[17OH2]', 
                      '[Zn-2]', '[Ca]', '[nH]', '[O-]', '[13C@H]', '[Po]', '[Ru+2]', '[NH+]', '[CH2-]', '[79BrH]', '[Au]',
                      '[Zn+2]', '[B-]', '[30Si]', '[C@H]', '[Te]', '[Ag+]', '[Eu]', '[P@]', '[PH]', '[HH]', '[73Ge]', 
                      '[197Au]', '[Al+3]', '[Mo]', '[PH+]', '[OH+]', '[Cs+]', '[183W]', '[Sn+2]', '[CH-]', '[n+]', '[2H]',
                      '[6Li]', '[2HH]', '[K+]', '[V+2]', '[33SH2]', '[Sb+]', '[PH4+]', '[203Hg]', '[93Nb]', '[51V]', 
                      '[PH2]', '[H]', '[S]', '[15OH2]', '[Cu+2]', '[13NH3]', '[Yb+3]', '[Pt]', '[NH3+]', '[65Zn]', '[OH2+]',
                      '[13C]', '[Ga]', '[O+]', '[Pt+]', '[Cd+2]', '[Gd+3]', '[As+]', '[Mn+2]', '[16OH2]', '[n-]', '[13CH2]', '[67Zn]', 
                      '[Ce]', '[203Tl]', '[28Si]', '[SH3+]', '[31Si]', '[Ba+2]', '[121Sb]', '[C+]', '[Mg+2]', '[Y+3]', '[32Si]', '[H-]', 
                      '[89Y]', '[N+]', '[Si]', '[63Cu]', '[32SH2]', '[Co+3]', '[13CH3]', '[Ni+3]', '[Al]', '[NH2+]',
                      ]
        # for i, token in enumerate(ion_tokens):
        tokenizer.add_tokens(ion_tokens)
        
        if hparams.tag:
            tokenizer.add_special_tokens({'additional_special_tokens': ['<SMILES>', '</SMILES>']})
        
        self.pretrained_model.resize_token_embeddings(len(tokenizer))
        self.fix_untrained_tokens(self.pretrained_model)
        self.tokenizer = tokenizer
        # self.pretrained_model.resize_token_embeddings(len(tokenizer))
    
    def preprocess_function(self, examples):
        inputs = examples["description"]
        targets = examples['smiles']
        if self.hparams.tag:
            targets = [f"<SMILES> {smiles} </SMILES>" for smiles in targets]
        
        cot_list = ["" for _ in range(len(targets))]
        cot_mode = map_cot_mode(self.hparams)
        cot_list = add_cot_to_target(examples, cot_list, cot_mode)
        
        if cot_list[0] == "":
            inputs = [f"{text} The SMILES of the molecule is: {smiles}. {self.tokenizer.eos_token}" for text, smiles in zip(inputs, targets)]
        else:
            cot_list =  [cot[1].lower() + cot[2:] for cot in cot_list]
            inputs = [f"{text} Then, {cot} The SMILES of the molecule is: {smiles}. {self.tokenizer.eos_token}" for text, cot, smiles in zip(inputs, cot_list, targets)]
        examples['text'] = inputs
        
        return examples
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='llama')
        parser.add_argument("--cot_mode_multiset", type=str, default='None')
        parser.add_argument("--cot_mode_fragment", action='store_true')
        parser.add_argument("--cot_mode_ring", action='store_true')
        parser.add_argument("--cot_mode_aromatic", action='store_true')
        parser.add_argument("--wandb_mode", type=str, default='disabled')
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_batch_size", type=int, default=2)
        parser.add_argument("--eval_batch_size", type=int, default=4)
        parser.add_argument("--gen_batch_size", type=int, default=32)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--epochs", type=int, default=10)
        # parser.add_argument("--task", type=str, default='', choices=['', '-caption2smiles'])
        parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
        parser.add_argument('--max_length', type=int, default=512)
        parser.add_argument('--test', action='store_false')
        parser.add_argument('--run_id', type=str, default='')
        parser.add_argument('--tag', action='store_false')
        parser.add_argument('--pretrain_model_id', type=str, default='')

        return parser

class WandbLlamaProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, model, hparams):
        super(WandbLlamaProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)
        self.model = model

    def on_evaluate(self, args, state, control, **kwargs):
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Llama Evaluation")
            # generate predictions
            
            run_name = map_cot_mode(self.hparams)
            if run_name == "":
                input_prompt = [f"{des} The SMILES of the molecule is: " for des in self.test_dataset['description']]
                if self.hparams.tag:
                    input_prompt = [f"{des} <SMILES> " for des in input_prompt]
            else:
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
            
            file_name = f'predictions/ft_cot_llama/{self.hparams.architecture}{run_name}.txt'

            description_list = self.test_dataset['description']
            gt_smiles = self.test_dataset['smiles']
            if hparams.tag:
                predicted_smiles = [dp[dp.find("<SMILES> "):].split(' ')[0][len("<SMILES> "):] if (dp.find("<SMILES>") > -1) else " " for dp in decoded_preds]
            else:
                predicted_smiles = [dp[dp.find("The SMILES of the molecule is: "):].split('.')[0][len("The SMILES of the molecule is: "):] if dp.find("The SMILES of the molecule is:") > -1 else " " for dp in decoded_preds]
            predicted_smiles = [smi[:smi.find('</SMILES>')] if smi.find('</SMILES>') > -1 else smi for smi in predicted_smiles]
            predicted_smiles = [smi if len(smi)>0 else " " for smi in predicted_smiles]
            predicted_smiles = [smi.replace(" ", "") for smi in predicted_smiles]
            
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
    run_name = map_cot_mode(hparams)

    # for hugging face login
    HfFolder.save_token('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')
    

    if hparams.run_id == '' or hparams.pretrain_model_id != '':
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-llama', mode=hparams.wandb_mode,
               group='ft_cot')
    else:
        wandb.init(project='mol2text', name=f'{hparams.architecture}{run_name}-ft-llama', mode=hparams.wandb_mode,
               group='ft_cot', resume='must', id=hparams.run_id)
    
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
        save_strategy='epoch',
        # warmup_ratio=0.1,
        # warmup_steps=1000
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
    
    if hparams.run_id == '' and hparams.pretrain_model_id == '':
        trainer.train()
    else:
        if hparams.pretrain_model_id != '':
            file_path = sorted([dI for dI in os.listdir(f'output/{hparams.pretrain_model_id}') if os.path.isdir(os.path.join(f'output/{hparams.pretrain_model_id}',dI))])[-1]
            trainer.train(resume_from_checkpoint=f"output/{hparams.pretrain_model_id}/{file_path}")
        else:
            file_path = sorted([dI for dI in os.listdir(f'output/{hparams.run_id}') if os.path.isdir(os.path.join(f'output/{hparams.run_id}',dI))])[-1]
            # need to check
            # trainer._load_optimizer_and_scheduler(f"output/{hparams.run_id}/{file_path}")
            trainer.train(resume_from_checkpoint=f"output/{hparams.run_id}/{file_path}")
    
    wandb.finish()
    