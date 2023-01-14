alg=supervised
batch_size=256
level=species
## unused args
kd_T=1.0
alpha=1.0
warmup=1
climit=25

## Supervised baseline 
MoCo=false
unlabel=in
for task in semi_inat; do
  for init in imagenet; do

    if [ ${init} == scratch ]
    then
      ## From scratch ##
      init=scratch
      num_iter=50000
      lr=3e-2
      wd=3e-3
    elif [ ${init} == imagenet ]
    then
      ## From ImageNet ##
      init=imagenet
      num_iter=50000
      lr=3e-3
      wd=1e-4
    elif [ ${init} == inat ]
    then
      ## From iNat ##
      init=inat
      num_iter=15000
      lr=1e-3
      # lr=3e-3
      wd=1e-4
    fi

    exp_dir=${task}_Supervised_climit_${climit}_${level}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
    # exp_dir=${task}_Supervised_${level}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
    echo "${exp_dir}"
    out_path=slurm_out_0504/${exp_dir}
    err_path=slurm_err_0504/${exp_dir}
    export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup climit
    # sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_train.sbatch
    python run_train.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
                        --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
                        --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --class_limit ${climit} --level ${level}

  done
done
