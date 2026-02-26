<h1 align="center">Structural Reasoning Improves Molecular Understanding of LLM </h1>

[![arXiv](https://img.shields.io/badge/arXiv-2410.05610-b31b1b.svg)](https://arxiv.org/abs/2410.05610)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

<div align="center">
  <a href="https://yunhuijang.github.io">Yunhui Jang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sites.google.com/view/jaehyungkim">Jaehyung Kim</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sungsoo-ahn.github.io">Sungsoo Ahn</a><sup>1</sup>
  <br>
  <sup>1</sup>KAIST &emsp; <sup>2</sup>Yonsei University
</div>

<h3 align="center">[<a href="https://arxiv.org/abs/2410.05610">arXiv</a>]</h3>

<b>Summary</b>: MSR framework explicitly incorporates key structural features into LLM reasoning, supporting both known target molecules (with explicit structure info) and unknown target molecules (through structural feature guidance).

## Installation

```bash
git clone https://github.com/yunhuijang/MSR.git
cd MSR
conda env create -f environment.yaml
conda activate msr
```

## Checkpoints

We provide the following fine-tuned checkpoints:

| Model | Description | Path |
|-------|-------------|------|
| ChemT5-small-analytic | Molecule-to-text model | `output/chemt5-small-analytic/` |
| ChemT5-small-t2m-reason | Text-to-molecule reasoning model | `output/chemt5-small-t2m-reason/` |
| ChemT5-small-t2m-answer | Text-to-molecule answer model | `output/chemt5-small-t2m-answer/` |

## Finetuning

### Analytic reasoning

```bash
python model/one_stage_generator_mol2text.py \
--architecture multitask-text-and-chemistry-t5-small-standard \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral-weight-name \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id GT4SD \
--weight_decay 0 \
--learning_rate 6e-4 \
--warmup_ratio 0.1 \
--check_val_every_n_epoch 20 \
--lr_scheduler_type linear \
--max_length 820 \
--generation_mode \
--max_new_tokens 512
```

### Synthethic reasoning

#### Reasoning module
```bash
python model/reasoning_generator.py \
--architecture multitask-text-and-chemistry-t5-small-standard \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral-weight-name \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id GT4SD \
--max_length 820 \
--generation_mode \
--max_new_tokens 256 \
--check_val_every_n_epoch 20 \
--weight_decay 0 \
--learning_rate 6e-4 \
--warmup_ratio 0 \
--lr_scheduler_type linear

```


#### Answering module
```bash
python model/answer_generator.py \
--architecture multitask-text-and-chemistry-t5-small-standard \
--cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral-weight-name \
--select_cot_mode chain-aromatic-con_ring_name-func_simple-chiral \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id GT4SD \
--max_length 820 \
--generation_mode \
--max_new_tokens 512 \
--check_val_every_n_epoch 20 \
--weight_decay 0 \
--learning_rate 6e-4 \
--warmup_ratio 0 \
--lr_scheduler_type linear \
--is_iterative
```
### Arguments

#### Common arguments (all scripts)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--architecture` | `str` | `multitask-text-and-chemistry-t5-small-standard` | Model architecture (see choices below) |
| `--cot_mode` | `str` | varies | Structural features for chain-of-thought reasoning. Combine multiple features with `-` (see choices below) |
| `--wandb_mode` | `str` | `disabled` | WandB logging mode (e.g., `online`, `disabled`) |
| `--learning_rate` | `float` | varies | Learning rate |
| `--train_batch_size` | `int` | varies| Training batch size per device |
| `--eval_batch_size` | `int` | varies | Evaluation batch size per device |
| `--weight_decay` | `float` | varies | Weight decay for optimizer |
| `--epochs` | `int` | `100` | Number of training epochs |
| `--check_val_every_n_epoch` | `int` | varies | Run validation every N epochs |
| `--max_length` | `int` | `512` | Maximum input sequence length |
| `--max_new_tokens` | `int` | `512` | Maximum number of tokens to generate |
| `--model_id` | `str` | `GT4SD` | HuggingFace model ID. Choices: `laituan245`, `QizhiPei`, `GT4SD` |
| `--dataset_name` | `str` | `lm` | Dataset to use. Choices: `molt5` (ChEBI-20), `lm` (LPM-24) |
| `--warmup_ratio` | `float` | `0` | Warmup ratio for learning rate scheduler |
| `--lr_scheduler_type` | `str` | `linear` | Learning rate scheduler type |

#### Answer module arguments (`answer_generator.py` only)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--select_cot_mode` | `str` | `aromatic-con_ring_name-func_simple` | Subset of CoT features to use from reasoning output |
| `--is_iterative` | flag | `False` | matching ratio-
based rejection sampling with beam search |
| `--num_iter` | `int` | `5` | Number of beam search (used with `--is_iterative`) |

#### `--architecture` choices

| Value | Description |
|-------|-------------|
| `multitask-text-and-chemistry-t5-small-standard` | ChemT5-small (default) |
| `multitask-text-and-chemistry-t5-base-standard` | ChemT5-base |
| `molt5-small` | MolT5-small |
| `molt5-base` | MolT5-base |
| `molt5-large` | MolT5-large |

#### `--cot_mode` features

Combine features with `-` separator (e.g., `chain-aromatic-func_simple`).

| Feature | Description |
|---------|-------------|
| `multiset_formula` | Molecular formula as multiset |
| `chain` | Carbon chain structure |
| `aromatic` | Aromatic ring information |
| `con_ring_name` | Ring names |
| `func_simple` | Functional groups |
| `chiral` | Chirality information |


## Datasets

- [ChEBI-20](https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data) 
- [LPM-24](https://huggingface.co/language-plus-molecules) 

## Citation

```bibtex
@inproceedings{jang2025msr,
    title={Structural Reasoning Improves Molecular Understanding of LLM},
    author={Jang, Yunhui and Kim, Jaehyung and Ahn, Sungsoo},
    booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2025}
}
```
