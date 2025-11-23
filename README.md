# TinyRecursiveMamba: TRM with SSMs

This codebase tries to experiment with SSM layers for recursive reasoning on tasks like ARC-AGI. Note that we don't experiment with the ARC task but instead use Sudoku-Extreme due to computational constraints.

### Disclaimers

- For ease of implementation, we do not used optimised cuda kernels for the SSM that one might find in the `mamba-ssm` package. SSM implementation focuses of minimalism, the ability to test and iterate fast, and readability.

- We use `torch.optim.AdamW` to train the model instead of `AdamATan2` due to installation difficulties.

### TODOs

- [x] Mamba SSM block modelling
- [x] Insert into TRM model and config
- [x] Prepare Sudoku-Extreme data
- [ ] Test forward pass of SSM block
- [ ] Test backward pass of SSM block
- [ ] Decide configs to run for attetion baseline
- [ ] Decide configs to run for SSM variant
- [ ] Decide ablation studies

### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements (except adam-atan2 for our case)
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments
```

## Experiments

### Sudoku-Extreme (assuming 1 L40S GPU, comparison done with attention variant):

```bash
# runtime < 36 hours

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```


## References

This repository is built on top of the <a href="https://github.com/SamsungSAILMontreal/TinyRecursiveModels">original TRM source code</a> with references from <a href="https://github.com/johnma2006/mamba-minimal">this nice and minimalist modelling of Mamba</a>. If you find our work useful, please consider citing TRM, HRM, and Mamba

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks},
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871},
}

@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734},
}

@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
