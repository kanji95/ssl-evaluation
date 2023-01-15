task=semi_inat
batch_size=60
level=species
# only used for distillation
kd_T=1.0
alpha=1.0
# data_root=/ssd_scratch/cvit/kanishk/inaturalist_2019/
data_root=/media/newhd/inaturalist_2019/
# only used for PL
warmup=1
climit=50

## Choose the algorithm
# alg=sup_hie
# alg=PL_hie
alg=MoCo_hie
# alg=ST_hie
# alg=MoCoST_hie
# alg=transfer
# level=species

################################
#### Supervised + hierarchy ####
################################
if [ ${alg} == sup_hie ]
then
alg=hierarchy
MoCo=false
# for level in genus kingdom phylum class order family species; do
for level in species; do
  for unlabel in in; do
  # for unlabel in in inout; do
    for init in imagenet; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=100000
        lr=3e-3
        wd=1e-3
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
        num_iter=50000
        lr=3e-3
        wd=1e-4
      fi

      ## only species loss for labeled data
      exp_dir=${task}_Supervised_hie_climit_${climit}_${level}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}
      # export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level climit data_root
      # sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch
      CUDA_VISIBLE_DEVICES=0,1,2 python run_train_hierarchy.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
                                    --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
                                    --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --level ${level} --class_limit ${climit} --data_root ${data_root}

    done
  done
done

################################
######## PL + hierarchy ########
################################
elif [ ${alg} == PL_hie ]
then
alg=PL_hierarchy
MoCo=false
kd_T=1.0
alpha=1.0
warmup=1
unlabel=inout
for level in species; do
  for unlabel in in; do
    for init in imagenet; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=100000
        warmup=10000
        lr=3e-2
        wd=1e-4
        threshold=0.95
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=50000
        warmup=5000
        # lr=1e-3
        lr=3e-3
        wd=1e-4
        threshold=0.85
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=50000
        warmup=5000
        # lr=1e-3
        lr=3e-3
        wd=1e-4
        threshold=0.95
      fi

      exp_dir=${task}_PL_hie_climit_${climit}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}

      # export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level climit
      # sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch
      CUDA_VISIBLE_DEVICES=0,1,2 python run_train_hierarchy.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
                                    --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
                                    --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --level ${level} --class_limit ${climit} --data_root ${data_root}

    done
  done
done

##################################
######## MoCo + hierarchy ########
##################################
elif [ ${alg} == MoCo_hie ]
then
alg=hierarchy
MoCo=true
unlabel=inout
for level in species; do
  for unlabel in in; do
    for init in imagenet; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=100000
        lr=3e-3
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
        num_iter=50000
        lr=1e-2
        wd=1e-4
      fi

      exp_dir=${task}_MoCo_hie_climit_${climit}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}
      # export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level climit
      # sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch
      CUDA_VISIBLE_DEVICES=0,1,2 python run_train_hierarchy.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
                                    --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
                                    --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --level ${level} --class_limit ${climit} --data_root ${data_root}

    done
  done
done

################################
######## ST + hierarchy ########
################################
elif [ ${alg} == ST_hie ]
then
alg=distill_hierarchy
MoCo=false
kd_T=1.0
alpha=0.7
warmup=1
unlabel=inout
for level in species; do
  for unlabel in in; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=150000
        lr=3e-2
        wd=1e-3
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
        num_iter=50000
        lr=3e-3
        wd=1e-4
      fi

      exp_dir=${task}_ST_hie_climit_${climit}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}

      # export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level climit
      # sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch
      CUDA_VISIBLE_DEVICES=0,1,2 python run_train_hierarchy.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
                                    --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
                                    --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --level ${level} --class_limit ${climit} --data_root ${data_root}

    done
  done
done

####################################
######## MoCoST + hierarchy ########
####################################
elif [ ${alg} == MoCoST_hie ]
then
alg=distill_hierarchy
MoCo=true
kd_T=1.0
alpha=0.7
warmup=1
unlabel=inout
for level in phylum; do
  for unlabel in in inout; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=150000
        lr=3e-2
        wd=1e-3
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
        num_iter=50000
        lr=3e-3
        wd=1e-4
      fi

      exp_dir=${task}_MoCoST_hie_climit_${climit}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}

      # export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level climit
      # sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch
      CUDA_VISIBLE_DEVICES=0,1,2 python run_train_hierarchy.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
                                    --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
                                    --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --level ${level} --class_limit ${climit} --data_root ${data_root}

    done
  done
done 

fi
