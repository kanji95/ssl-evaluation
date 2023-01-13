#!/bin/bash

#SBATCH -n 38
#SBATCH -N 1
#SBATCH --gres=gpu:4
##SBATCH --mem-per-cpu=12288
#SBATCH --time=4-00:00:00
#SBATCH --job-name=SSL
#SBATCH -w gnode091

# module load python/3.6.8

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate tracking

module load u18/cuda/11.6 # cuda/11.0
module load u18/cudnn/8.4.0-cuda-11.6 # cudnn/8-cuda-11.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

# set -e

# rm -rf /ssd_scratch/cvit/kanishk/carla_data

# if [ ! -d "/ssd_scratch/cvit/kanishk/imagenet/" ]; then
#     echo "Copying VAL Imagenetg"
#     mkdir -p /ssd_scratch/cvit/kanishk/imagenet/
#     scp -r kanishk@ada:/share3/kanishk/val_imagenet.tar.gz /ssd_scratch/cvit/kanishk/
#     tar -xf /ssd_scratch/cvit/kanishk/val_imagenet.tar.gz -C /ssd_scratch/cvit/kanishk/
#     rm -rf /ssd_scratch/cvit/kanishk/val_imagenet.tar.gz
# 
#     scp -r kanishk@ada:/share3/kanishk/imagenet-a.tar /ssd_scratch/cvit/kanishk/
#     tar -xf /ssd_scratch/cvit/kanishk/imagenet-a.tar -C /ssd_scratch/cvit/kanishk/
#     rm -rf /ssd_scratch/cvit/kanishk/imagenet-a.tar
# 
#     scp -r kanishk@ada:/share3/kanishk/imagenet-o.tar /ssd_scratch/cvit/kanishk/
#     tar -xf /ssd_scratch/cvit/kanishk/imagenet-o.tar -C /ssd_scratch/cvit/kanishk/
#     rm -rf /ssd_scratch/cvit/kanishk/imagenet-o.tar
# fi

# if [ ! -d "/ssd_scratch/cvit/kanishk/imagenet" ]; then
#     rm -rf /ssd_scratch/cvit/kanishk/
#     mkdir -p /ssd_scratch/cvit/kanishk/
#     echo "Copying Imagenet"
#     scp -r kanishk@ada:/share1/dataset/Imagenet2012/Imagenet-orig.tar /ssd_scratch/cvit/kanishk/
#     echo "Extracting Imagenet"
#     tar -xf /ssd_scratch/cvit/kanishk/Imagenet-orig.tar -C /ssd_scratch/cvit/kanishk/
#     cd /ssd_scratch/cvit/kanishk/
#     mv Imagenet-orig/ILSVRC2012_img_* ./
#     rm -rf Imagenet-orig*
#     cp ~/imagenet/extract_ILSVRC.sh ./
#     echo "Creating Train & Val Folders"
#     sh extract_ILSVRC.sh
#     cd ~/post_hoc_correction
# fi

if [ ! -d "/ssd_scratch/cvit/kanishk/inaturalist_2019/" ]; then
    mkdir -p /ssd_scratch/cvit/kanishk/inaturalist_2019
    scp -r kanishk@ada:/share3/kanishk/inaturalist_2019.tar.gz /ssd_scratch/cvit/kanishk/
    tar -xf /ssd_scratch/cvit/kanishk/inaturalist_2019.tar.gz -C /ssd_scratch/cvit/kanishk/
    rm -rf /ssd_scratch/cvit/kanishk/inaturalist_2019.tar.gz
fi

export PYTHONUNBUFFERED=TRUE

# bash experiments/train/tieredimagenet/cross-entropy.sh
# bash experiments/train/inat/cross-entropy.sh

# python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed0 --seed 0 --taxonomy genus --pretrained
# python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed1 --seed 1 --taxonomy genus --pretrained
# python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed2 --seed 2 --taxonomy genus --pretrained
# python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed3 --seed 3 --taxonomy genus --pretrained
# python main.py --start training --arch custom_resnet18 --batch-size 256 --epochs 100 --loss cross-entropy --optimizer custom_sgd --data inaturalist19-224 --workers 16 --output out/inat/cross-entropy/genus/seed4 --seed 4 --taxonomy genus --pretrained

./exp_sup.sh
