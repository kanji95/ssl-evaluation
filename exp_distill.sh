alg=distill
batch_size=64
## unused args
warmup=1

## Self-Training ##
for unlabel in in; do
  for MoCo in true; do
    kd_T=1.0
    alpha=0.7
    # for task in semi_aves semi_fungi; do
    for task in semi_fungi; do
      for init in scratch imagenet; do

        if [ ${init} == scratch ]
        then
          ## From scratch ##
          init=scratch
          num_iter=50000
          lr=3e-2
          wd=1e-3
        elif [ ${init} == imagenet ]
        then
          ## From ImageNet ##
          init=imagenet
          num_iter=50000
          lr=1e-2
          wd=1e-4     
        elif [ ${init} == inat ]
        then
          ## From iNat ##
          init=inat
          num_iter=10000
          lr=1e-2
          wd=1e-4
        fi

        if [ ${MoCo} == true ]
        then
          exp_dir=${task}_${alg}_${init}_MoCo_${unlabel}_${lr}_${wd}_${num_iter}
        else
          exp_dir=${task}_${alg}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
        fi
        echo "${exp_dir}"
        out_path=slurm_out/${exp_dir}
        err_path=slurm_err/${exp_dir} 
        export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup
        # sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_train.sbatch
        python run_train.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
                            --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
                            --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T}

      done
    done
  done
done