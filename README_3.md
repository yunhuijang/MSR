<h1 align="center">Structural Reasoning Improves Molecular Understanding of LLM </h1>

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

<div align="center">
  <a href="https://github.com/yunhuijang">Yunhui Jang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="">Jaehyung Kim</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="">Sungsoo Ahn</a><sup>1</sup>
  <br>
  <sup>1</sup>KAIST &emsp; <sup>2</sup>Yonsei University
</div>

<h3 align="center">[<a href="https://arxiv.org/abs/2025.xxxxx">arXiv</a>]</h3>

<b>Summary</b>: MSR framework explicitly incorporates key structural features into LLM reasoning, supporting both known target molecules (with explicit structure info) and unknown target molecules (through structural feature guidance). 


## Installation

```bash
# Clone the repository
git clone https://github.com/yunhuijang/MSR.git
cd MSR

# Create conda environment
conda env create -f environment.yaml
conda activate msr
```


## Pre-trained Checkpoints

| Model | Description | Path |
|-------|-------------|------|
| MolT5-base-m2t | Molecule-to-text base model | `output/molt5-base-m2t/` |
| ChemT5-base-t2m-reason | Text-to-molecule reasoning model | `output/chemt5-base-t2m-reason/` |
| ChemT5-base-t2m-answer | Text-to-molecule answer model | `output/chemt5-base-t2m-answer/` |

## Usage

### Fine-tuning

The framework supports different chain-of-thought (CoT) modes:

```bash
# Generalist fine-tuning
bash script/generalist/train.sh --cot_mode [none|reason|answer]

# Specialist fine-tuning
bash script/specialist/train.sh --cot_mode [none|reason|answer]
```

**CoT Mode Options:**
- `none`: Standard fine-tuning without chain-of-thought
- `reason`: Fine-tuning with structural reasoning chain
- `answer`: Fine-tuning with answer-focused chain

### Evaluation

```bash
# Run evaluation
bash script/evaluate.sh --model_path output/your_model/
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{jang2025msr,
    title={Structural Reasoning Improves Molecular Understanding of LLM},
    author={Jang, Yunhui and Kim, Jaehyung and Ahn, Sungsoo},
    booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2025}
}
```

## Authors

- [Yunhui Jang](https://github.com/yunhuijang)
- Jaehyung Kim
- Sungsoo Ahn

## Acknowledgments

This project is built upon the [MolT5](https://github.com/blender-nlp/MolT5) project. We thank the authors for their excellent work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
