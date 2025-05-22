# Structural Reasoning Improves Molecular Understanding of LLM

In this repository, we implement the paper: Structural Reasoning Improves Molecular Understanding of LLM (MSR) in ACL 2025.

### Checkpoints
We provide a few initial checkpoints and plan to release additional ones in future updates.

+ [MolT5-base-m2t] output/chemt5-small-m2t
+ [ChemT5-base-t2m-reason] output/chemt5-small-t2m-reason
+ [ChemT5-base-t2m-answer] output/chemt5-small-t2m-answer

### Finetuning 
You can use the script in script/ for fine-tuning with our MSR.
For generalists, you can use the scripts in script/generalist/ and for specailists, you can use the scripts in scrip/specialist.
The cot_mode option determines the structural information included in MSR.
Note that you need to use your openai and huggingface key.


### Datasets
 - [ChEBI-20](https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data) (txt format)


Our code is based on https://github.com/blender-nlp/MolT5.
Copyright (c) 2023, blender-nlp
