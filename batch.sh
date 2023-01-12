# MoCo + Self-Supervised
CUDA_VISIBLE_DEVICES=0,1,2 python run_train.py --task semi_inat --init imagenet --alg supervised --unlabel in --num_iter 200000 --lr 1e-3 --wd 1e-3 --exp_dir Moco_self_supervised/cl_50 --MoCo true --val_freq 5000 --model custom_resnet18 --print_freq 1000 --class_limit 50

# MoCo + Self-Training
CUDA_VISIBLE_DEVICES=0,1,2 python run_train.py --task semi_inat --init imagenet --alg distill --unlabel in --num_iter 200000 --lr 1e-2 --wd 1e-4 --exp_dir Moco_self_training/cl_50 --MoCo true --val_freq 5000 --path_t results/Moco_self_supervised/cl_50/checkpoints/checkpoint.pth.tar --warmup 1 --kd_T 1.0 --alpha 0.7 --model custom_resnet18 --print_freq 1000 --class_limit 50

# Curriculum Pseudo-Labeling
CUDA_VISIBLE_DEVICES=0,1,2 python run_train.py --task semi_inat --init imagenet --alg supervised --unlabel in --num_iter 200000 --lr 1e-3 --wd 1e-4 --exp_dir CPL/cl_50 --MoCo false --val_freq 5000 --model custom_resnet18 --print_freq 1000 --kd_T 1.0 --alpha 0.7 --warmup 1 --class_limit 50

# Pseudo-Labeling
CUDA_VISIBLE_DEVICES=0,1,2 python run_train.py --task semi_inat --init imagenet --alg PL --unlabel in --num_iter 200000 --lr 1e-3 --wd 1e-4 --exp_dir PL/cl_50 --MoCo false --val_freq 5000 --model custom_resnet18 --print_freq 1000 --kd_T 1.0 --alpha 0.7 --warmup 10000 --threshold 0.95 --class_limit 50