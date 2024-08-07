# HopMAE: Self-Supervised Graph Masked Auto-Encoders from a Hop Perspective

Implementation of ICIC2024 paper [HopMAE: Self-Supervised Graph Masked Auto-Encoders from a Hop Perspective](https://doi.org/10.1007/978-981-97-5666-7_29)

## Requirements

This repository has been tested with the following packages:
- Python == 3.9.17
- PyTorch == 2.0.1
- DGL == 2.0.0

## Notification
- Before running [main.py](main.py), please preprocess the dataset to generate the required hop files by specifying the dataset variable in [save.py](save.py) and run it.
- Please refer to [main.py](main.py) for the full hyper-parameters.

## How to Run
```
# cora
python main.py --dataset cora --hops 10 --alpha 0.1 --hidden_dim 128 --ffn_dim 32 --n_heads 4 --dropout 0.75 --attention_dropout 0.4 --n_encoder_layers 1 --n_decoder_layers 1 --mask_rate 0.7 --remask_rate 0.4 --device 0 --batch_size 30000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation gelu --pool_type max --pre_optim_type radam --pre_max_epoch 1500 --pre_lr 0.001 --pre_weight_decay 0.005 --tau 100 --optim_type adamw --max_epoch 300 --lr 0.04 --weight_decay 0.0003
# citeseer
python main.py --dataset citeseer --hops 3 --alpha 0.0001 --hidden_dim 128 --ffn_dim 32 --n_heads 2 --dropout 0.75 --attention_dropout 0.3 --n_encoder_layers 1 --n_decoder_layers 1 --mask_rate 0.55 --remask_rate 0.25 --device 0 --batch_size 30000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation relu --pool_type mean --pre_optim_type radam --pre_max_epoch 1500 --pre_lr 0.004 --pre_weight_decay 0.005 --tau 50 --optim_type sgd --max_epoch 300 --lr 0.04 --weight_decay 0.0001
# pubmed
python main.py --dataset pubmed --hops 3 --alpha 5 --hidden_dim 128 --ffn_dim 64 --n_heads 4 --dropout 0.7 --attention_dropout 0.5 --n_encoder_layers 2 --n_decoder_layers 3 --mask_rate 0.7 --remask_rate 0.1 --device 1 --batch_size 30000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation elu --pool_type mean --pre_optim_type adam --pre_max_epoch 1500 --pre_lr 0.002 --pre_weight_decay 0.0005 --tau 50 --optim_type radam --max_epoch 900 --lr 0.01 --weight_decay 0.0003
# ogbn-arxiv
python main.py --dataset ogbn-arxiv --hops 15 --alpha 0.005 --hidden_dim 512 --ffn_dim 256 --n_heads 2 --dropout 0.3 --attention_dropout 0.5 --n_encoder_layers 3 --n_decoder_layers 3 --mask_rate 0.2 --remask_rate 0.5 --device 1 --batch_size 5000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation relu --pool_type mean --pre_optim_type adadelta --pre_max_epoch 100 --pre_lr 0.05 --pre_weight_decay 0.0005 --tau 20 --optim_type adamw --max_epoch 1000 --lr 0.5 --weight_decay 0
# photo
python main.py --dataset photo --hops 3 --alpha 5 --hidden_dim 128 --ffn_dim 32 --n_heads 8 --dropout 0.5 --attention_dropout 0.4 --n_encoder_layers 2 --n_decoder_layers 1 --mask_rate 0.3 --remask_rate 0.1 --device 1 --batch_size 30000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation elu --pool_type mean --pre_optim_type adamw --pre_max_epoch 1500 --pre_lr 0.0009 --pre_weight_decay 0.0005 --tau 100 --optim_type adam --max_epoch 300 --lr 0.02 --weight_decay 0.0005
# computer
python main.py --dataset computer --hops 3 --alpha 0.01 --hidden_dim 128 --ffn_dim 32 --n_heads 2 --dropout 0.65 --attention_dropout 0.4 --n_encoder_layers 3 --n_decoder_layers 2 --mask_rate 0.6 --remask_rate 0.4 --device 0 --batch_size 30000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation prelu --pool_type mean --pre_optim_type adamw --pre_max_epoch 1500 --pre_lr 0.001 --pre_weight_decay 0.005 --tau 100 --optim_type adam --max_epoch 600 --lr 0.03 --weight_decay 0.0001
# cs
python main.py --dataset cs --hops 1 --alpha 0.005 --hidden_dim 128 --ffn_dim 64 --n_heads 4 --dropout 0.4 --attention_dropout 0.4 --n_encoder_layers 1 --n_decoder_layers 2 --mask_rate 0.6 --remask_rate 0.3 --device 0 --batch_size 10000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation relu --pool_type mean --pre_optim_type adamw --pre_max_epoch 100 --pre_lr 0.0009 --pre_weight_decay 0.005 --tau 20 --optim_type sgd --max_epoch 600 --lr 0.04 --weight_decay 0.0005
# physics
python main.py --dataset physics --hops 1 --alpha 0.0001 --hidden_dim 128 --ffn_dim 32 --n_heads 8 --dropout 0.6 --attention_dropout 0.5 --n_encoder_layers 1 --n_decoder_layers 1 --mask_rate 0.7 --remask_rate 0.2 --device 0 --batch_size 5000 --seeds 0 1 2 3 4 5 6 7 8 9 --activation gelu --pool_type mean --pre_optim_type adam --pre_max_epoch 100 --pre_lr 0.002 --pre_weight_decay 0.0005 --tau 20 --optim_type adam --max_epoch 300 --lr 0.05 --weight_decay 0.0003

```

## Acknowledgements

The code is implemented based on [GraphMAE](https://github.com/THUDM/GraphMAE) and [NAGphormer](https://github.com/JHL-HUST/NAGphormer).

## Citation

If you find this work is helpful to your research, please consider citing our paper:

```
@InProceedings{10.1007/978-981-97-5666-7_29,
author="Shi, Chenjunhao
and Li, Jin
and Zhuang, Jianzhi
and Yao, Xi
and Huang, Yisong
and Fu, Yang-Geng",
editor="Huang, De-Shuang
and Zhang, Chuanlei
and Pan, Yijie",
title="HopMAE: Self-supervised Graph Masked Auto-Encoders from a Hop Perspective",
booktitle="Advanced Intelligent Computing Technology and Applications",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="343--355",
abstract="With increasing popularity and larger real-world applicability, graph self-supervised learning (GSSL) can significantly reduce labeling costs by extracting implicit input supervision. As a promising example, graph masked auto-encoders (GMAE) can encode rich node knowledge by recovering the masked input components, e.g., features or edges. Despite their competitiveness, existing GMAEs focus only on neighboring information reconstruction, which totally ignores distant multi-hop semantics and thus fails to capture global knowledge. Furthermore, many GMAEs cannot scale on large-scale graphs since they suffer from memory bottlenecks with unavoidable full-batch training. To address these challenges and facilitate ``high-level'' discriminative semantics, we propose a simple yet effective framework (i.e., HopMAE) to encourage hop-perspective semantic interactions by adopting multi-hop input-rich reconstruction while supporting mini-batch training. Despite the rationales of the above designs, we still observe some limitations (e.g., sub-optimal generalizability and training instability), potentially due to the implicit gap between the task-triviality and input-richness of reconstruction. Therefore, to alleviate task-triviality and fully unleash the potential of our framework, we further propose a combined fine-grained loss function, which generalizes the existing ones and significantly improves the difficulties of reconstruction tasks, thus naturally alleviating over-fitting. Extensive experiments on eight benchmarks demonstrate that our method comprehensively outperforms many state-of-the-art counterparts.",
isbn="978-981-97-5666-7"
}
```
