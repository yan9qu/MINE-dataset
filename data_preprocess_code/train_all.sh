#!/bin/sh
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.1  --dropout 0.1
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.2  --dropout 0.2
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.3  --dropout 0.3
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.4  --dropout 0.4
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.5  --dropout 0.5
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.6  --dropout 0.6
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.7  --dropout 0.7
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.8  --dropout 0.8
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2316 main_mlc.py --output /data/sqhy_model/uncertainty/Intentonomy/dropout/0.9  --dropout 0.9
