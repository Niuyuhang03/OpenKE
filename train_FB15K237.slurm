#!/bin/bash
#SBATCH -o train_FB15K237.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J train_FB15K237             # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
python train_FB15K237.py
