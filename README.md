# SchemaWalk

This repository contains a PyTorch implementation of SchemaWalk, as described in the paper:

Shixuan Liu, Changjun Fan, Kewei Cheng, Yunfei Wang, Peng Cui, Yizhou Sun, Zhong Liu. [Inductive meta-path learning for schema-complex heterogeneous information networks](https://ieeexplore.ieee.org/abstract/document/10613499/). IEEE Transactions on Pattern Analysis and Machine Intelligence (2024). 

![Overview](https://github.com/shixuanliu-andy/SchemaWalk/tree/main/paper/overview.jpg)


## Overview

Heterogeneous Information Networks (HINs) are information networks with multiple types of nodes and edges. The concept of meta-path, i.e., a sequence of entity types and relation types connecting two entities, is proposed to provide the meta-level explainable semantics for various HIN tasks. Traditionally, meta-paths are primarily used for schema-simple HINs, e.g., bibliographic networks with only a few entity types, where meta-paths are often enumerated with domain knowledge. However, the adoption of meta-paths for schema-complex HINs, such as knowledge bases (KBs) with hundreds of entity and relation types, has been limited due to the computational complexity associated with meta-path enumeration. Additionally, effectively assessing meta-paths requires enumerating relevant path instances, which adds further complexity to the meta-path learning process.To address these challenges, we propose~\model, an inductive meta-path learning framework for schema-complex HINs. We represent meta-paths with schema-level representations to support the learning of the scores of meta-paths for varying relations, mitigating the need of exhaustive path instance enumeration for each relation. Further, we design a reinforcement-learning based path-finding agent, which directly navigates the network schema (i.e., schema graph) to learn policies for establishing meta-paths with high coverage and confidence for multiple relations. Extensive experiments on real data sets demonstrate the effectiveness of our proposed paradigm.

## System Requirements

Recommended software versions:

```bash
python==3.8
pytorch==1.12.1
numpy==1.23.4
scipy==1.3.1
```

## Training Instructions

### Multi-Relation Inductive Setting

Example for Yago/DBpedia:

```bash
python train.py --base_iterations 500 --beta 0.1 --hc_coff 10.0 --hidden_size 200 --reward_device auto --global_coverage --gpu 0
python train.py --base_iterations 500 --beta 0.05 --hc_coff 5.0 --hidden_size 400 --reward_device auto --global_coverage --gpu 0
```
### Multi-Relation Transductive Setting

Example for Yago/DBpedia:

```bash
python train.py --base_iterations 500 --beta 0.1 --hc_coff 10.0 --hidden_size 200 --reward_device auto --global_coverage --gpu 0 --transductive
python train.py --base_iterations 500 --beta 0.05 --hc_coff 5.0 --hidden_size 400 --reward_device auto --global_coverage --gpu 0 --transductive
```

### Per-Relation Inductive Setting

Example:

```bash
python train.py --gpu 0 --base_iterations 1000 --reward_device cpu --target_rel_index 1 --hc_coff 10 --beta 0.05
```

Adjust `target_rel_index` according to the index of the target relation in the relation set (see train.py).

## Trained Models and Meta-Paths

Some pre-trained models and outputted meta-paths are available in `./output/` and `./metapaths/`.


## Reference

If you find our code or paper useful, please cite:

```bibtex
@article{liu2024inductive,
  title={Inductive meta-path learning for schema-complex heterogeneous information networks},
  author={Liu, Shixuan and Fan, Changjun and Cheng, Kewei and Wang, Yunfei and Cui, Peng and Sun, Yizhou and Liu, Zhong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
