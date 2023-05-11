#!/bin/bash

trap "kill 0" EXIT

OLDIFS=$IFS

cuda_limit_per_gpu=6
num_devices=5
PORT=10000
e=0
cloud_path="files"
device_type="local"
dataset_name="cifar10"
model_name="conv5"
aps=100
seed=42

declare -a elems1=(
# Mobility | Cosine  | hierarchical | k2 | sigma | ap_option | train_batch_size
  "true false true 2 0.1 hierfavg 8" # HierFAVG
  "true true true 2 0.1 hierfavg 8" # MACFL
  "true true true 2 0.1 use_only_trained_aps 8" # MOHAWK
)

declare -a elems2=(
# IID? | Learning rate | Momentum
  "true 0.01 0.9" # IID
  "false 0.01 0.9" # Non-IID
)

local_ep=5
cloud_usr="usr"
cloud_pwd="pwd"
cloud_port="22"
cloud_cuda="cuda:0"
my_num_users=1000 # for splitting the dataset between my_num_users users

local_testing="false"
PORT_INCR=50

for elem2 in "${elems2[@]}"
do
  for elem1 in "${elems1[@]}"
  do
    rm -rf $cloud_path
    read -a tuple1 <<< "$elem1" # Mobility | Cosine  | hierarchical | k2 | sigma | ap_option | train_batch_size
    read -a tuple2 <<< "$elem2" # IID? | Learning rate | Momentum
    e=$((e+1))
    for r in 1
    do
      total_cuda=1

      data_iid="${tuple2[0]}"

      local_epochs="${local_ep}"

      for i in $(seq 0 $((num_devices-1)))
      do
        total_cuda=$((total_cuda+1))
        if [ $total_cuda -gt $((cuda_limit_per_gpu*3)) ]
        then
          cuda[$i]="cuda:3"
        elif [ $total_cuda -gt $((cuda_limit_per_gpu*2)) ]
        then
          cuda[$i]="cuda:2"
        elif [ $total_cuda -gt $((cuda_limit_per_gpu)) ]
        then
          cuda[$i]="cuda:1"
        else
          cuda[$i]="cuda:0"
        fi
        dev_local_epochs[$i]=$local_epochs
        dev_hw_types[$i]=$device_type
        ips[$i]="127.0.0.1"
        ports[$i]=$((PORT+i))
        model_names[$i]="${model_name}"
        train_batch_sizes[$i]="${tuple1[6]}"
      done
      cloud="cloud_cfg_exp${e}_run${r}.json"
      dev_config_filename="dev_cfg_exp${e}_run${r}.json"
      logfile="log_exp${e}_run${r}.csv"
      exec="sim_exp${e}_run${r}.bash"
      sim="execs/${exec}"
      echo $e, $r
      python simulation.py \
      --num_devices "${num_devices}" \
      --dev_local_epochs "${dev_local_epochs[*]}" \
      --dev_hw_types "${dev_hw_types[*]}" \
      --dev_config_filename ${dev_config_filename} \
      --ips "${ips[*]}" \
      --ports "${ports[*]}" \
      --cuda_names "${cuda[*]}" \
      --model_names "${model_names[*]}" \
      --train_batch_sizes "${train_batch_sizes[*]}" \
      --cloud_configs_filename ${cloud} \
      --log_file ${logfile} \
      --local_testing "${local_testing}" \
      --learning_rate "${tuple2[1]}" \
      --cloud_cuda ${cloud_cuda} \
      --dataset_name ${dataset_name} \
      --train_batch_size "${tuple1[6]}" \
      --data_iid "${data_iid}" \
      --num_users ${my_num_users} \
      --model_name "${model_name}" \
      --exec_name ${exec} \
      --cloud_usr ${cloud_usr} \
      --cloud_pwd ${cloud_pwd} \
      --cloud_port ${cloud_port} \
      --cloud_path ${cloud_path} \
      --momentum "${tuple2[2]}" \
      --mobility "${tuple1[0]}" \
      --mobility_devices "${my_num_users}" \
      --cosine "${tuple1[1]}" \
      --hierarchical "${tuple1[2]}" \
      --k2 "${tuple1[3]}" \
      --sigma "${tuple1[4]}" \
      --ap_option "${tuple1[5]}" \
      --aps "${aps}" \
      --myseed "${seed}"

      bash ${sim}
      PORT=$((PORT+PORT_INCR))
      sleep 20
    done
    unset k_sizes
    unset hw_type
    unset local_epochs
    unset cuda
  done
done
IFS=$OLDIFS
