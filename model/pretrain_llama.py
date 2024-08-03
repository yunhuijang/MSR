import transformers
import torch
from huggingface_hub.hf_api import HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import argparse
import wandb
import os
from datasets import Dataset
import pandas as pd
from rdkit import Chem


from model.one_stage_generator_llama import FineTuneTranslatorLlama, WandbPredictionProgressCallback

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class PreTrainLlama(FineTuneTranslatorLlama):
    def __init__(self, hparams):
        super(PreTrainLlama, self).__init__(hparams)

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
        
        self.pretrained_model = get_peft_model(pretrained_model, self.peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=hparams.max_length)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_0|>"})
            self.pretrained_model.config.pad_token_id = tokenizer.pad_token_id            
        
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
        self.pretrained_model.resize_token_embeddings(len(tokenizer))
        self.fix_untrained_tokens(self.pretrained_model)
        self.tokenizer = tokenizer
    

    
    def load_dataset(self, split):
        smiles_list_path = [os.path.join('resource/data', f'x00{i}.csv') for i in range(10)]
        dfs = [pd.read_csv(file_path) for file_path in smiles_list_path]
        df = pd.concat(dfs, ignore_index=True, copy=False)
        df = df[df['set']==split]
        smiles_list = df['smiles'].tolist()
            

        data_dict = {'smiles': smiles_list}
        dataset = Dataset.from_dict(data_dict)
        return dataset

    
    def preprocess_function(self, examples):
        inputs = examples['smiles']
        inputs = [input_ + self.tokenizer.eos_token for input_ in inputs]
        examples['text'] = inputs
        
        return examples
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--architecture", type=str, default='llama')
        parser.add_argument("--wandb_mode", type=str, default='disabled')
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_batch_size", type=int, default=2)
        parser.add_argument("--eval_batch_size", type=int, default=4)
        parser.add_argument("--gen_batch_size", type=int, default=32)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--epochs", type=int, default=200)
        # parser.add_argument("--task", type=str, default='', choices=['', '-caption2smiles'])
        parser.add_argument("--check_val_every_n_epoch", type=int, default=20)
        parser.add_argument('--max_length', type=int, default=100)
        parser.add_argument('--test', action='store_false')
        parser.add_argument('--run_id', type=str, default='')
        parser.add_argument('--num_samples', type=int, default=100)

        return parser

class WandPreTrainLlamaProgressCallback(WandbPredictionProgressCallback):
    def __init__(self, trainer, tokenizer, test_dataset, model, hparams):
        super(WandPreTrainLlamaProgressCallback, self).__init__(trainer, tokenizer, test_dataset, hparams)
        self.model = model

    def on_evaluate(self, args, state, control, **kwargs):
        if ((state.epoch + 1) % self.hparams.check_val_every_n_epoch == 0) or (state.epoch == 1):
            print("Start Llama Pretrain Evaluation")
            # generate predictions

            input_prompt = ["" for _ in range(hparams.num_samples)]
            
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
            
            file_name = f'predictions/pretrain_llama/{self.hparams.architecture}_{state.epoch}.txt'

            predicted_smiles = decoded_preds
            
            with open(f'{file_name}', 'w') as f:
                for ot in zip(predicted_smiles):
                    f.write(ot + '\n')
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    PreTrainLlama.add_args(parser)
    hparams = parser.parse_args()
    model = PreTrainLlama(hparams)
    if torch.cuda.is_available():
        model.to(device='cuda:0')
    else:
        model.to(device='cpu')
    print(model.device)

    # for hugging face login
    HfFolder.save_token('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')
    

    if hparams.run_id == '':
        wandb.init(project='mol2text', name=f'{hparams.architecture}-pre-llama', mode=hparams.wandb_mode,
               group='pretrain')
    else:
        wandb.init(project='mol2text', name=f'{hparams.architecture}-pre-llama', mode=hparams.wandb_mode,
               group='pretrain', resume='must', id=hparams.run_id)
    
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
        run_name=f'{hparams.architecture}-pre-llama',
        do_train=True,
        optim="paged_adamw_32bit",
        # load_best_model_at_end=True,
        save_strategy='steps',
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
        max_seq_length=hparams.max_length,
        dataset_text_field='text',
        packing=False
    )
    
    wandb_callback = WandPreTrainLlamaProgressCallback(trainer, model.tokenizer, model.test_dataset_tokenized, model=model.pretrained_model, hparams=hparams)
    
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
    