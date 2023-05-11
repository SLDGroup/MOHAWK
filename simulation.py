import argparse
import json
from os import path, mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--num_devices', type=int, default=3, help='Number of devices actually running on hardware')
parser.add_argument('--dev_local_epochs', nargs='+', default=[2, 2, 2], help='Local epochs for each device')
parser.add_argument('--dev_hw_types', nargs='+', default=["local", "local", "local"], help='Hardware type for each device')
parser.add_argument('--dev_config_filename', type=str, help='Configuration file name')
parser.add_argument('--ips', nargs='+', default=["127.0.0.1", "127.0.0.1", "127.0.0.1"], help='IPs for devices')
parser.add_argument('--ports', nargs='+', default=[9090, 9091, 9092], help='Ports for devices')
parser.add_argument('--cuda_names', nargs='+', default=['cuda:0', 'cuda:1', 'cpu'], help='Cuda_name for each device [cuda:number, cpu]')
parser.add_argument('--train_batch_sizes', nargs='+', default=[64, 32, 16], help='Train batch sizes for devices')
parser.add_argument('--model_names', nargs='+', default=['resnet_20', 'resnet_20', 'resnet_20'], help='Model name')

# Cloud configs
parser.add_argument('--cloud_configs_filename', type=str, help='File name for cloud configs')
parser.add_argument('--log_file', type=str, help='File for logging')
parser.add_argument('--myseed', type=int, help='Seed')
parser.add_argument('--local_testing', type=str, help='Test locally or not [true/false]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--cloud_cuda', type=str, default="cuda:0", help='Cloud cuda_name')
parser.add_argument('--dataset_name', type=str, default="cifar10", help='Dataset name')
parser.add_argument('--train_batch_size', type=int, default=128, help='Training batch size')
parser.add_argument('--data_iid', type=str, help='IID [true] or non-IID [false]')
parser.add_argument('--num_users', type=int, help='Total number of users')
parser.add_argument('--n_class', type=int, help='Classes per user')
parser.add_argument('--model_name', type=str, help='Model name')
parser.add_argument('--cloud_usr', type=str, help='Cloud username')
parser.add_argument('--cloud_pwd', type=str, help='Cloud password')
parser.add_argument('--cloud_path', type=str, help='Cloud path')
parser.add_argument('--cloud_port', type=int, default=22, help='Cloud port')
parser.add_argument('--momentum', type=float, default=0.9, help='Learning rate momentum')

# Mobility
parser.add_argument('--mobility', type=str, default='false', help='Enable mobility?')
parser.add_argument('--cosine', type=str, default='false', help='Use cosine if True, FedAvg if false')
parser.add_argument('--hierarchical', type=str, default='false', help='if using star topology or hierarchical')
parser.add_argument('--mobility_devices', type=int, default=100, help='number of mobility devices')
parser.add_argument('--k2', type=int, default=10, help='number of edge aggregations for AP')
parser.add_argument('--sigma', type=float, default=0.1, help='sigma for cosine similarity')
parser.add_argument('--ap_option', type=str, default='use_only_trained_aps', help='choose from [use_only_trained_aps, hierfavg]')
parser.add_argument('--aps', type=int, default=-1, help='-1 for all, AP >= 0 for how many APs we have')

# Bash executable configs
parser.add_argument('--exec_name', type=str, help='Bash executable name')

args = parser.parse_args()

# Bash configs
exec_name = args.exec_name

# CLOUD CONFIGS
cloud_configs_filename = args.cloud_configs_filename

log_file = args.log_file
seed = args.myseed
local_testing = args.local_testing
if local_testing == "true":
    local_testing = True
else:
    local_testing = False
learning_rate = args.learning_rate

cloud_cuda = args.cloud_cuda
dataset_name = args.dataset_name

train_batch_size = args.train_batch_size
data_iid = args.data_iid
if data_iid == "true":
    data_iid = True
else:
    data_iid = False
num_users = args.num_users
n_class = args.n_class
model_name = args.model_name

cloud_port = args.cloud_port
cloud_usr = args.cloud_usr
cloud_pwd = args.cloud_pwd
cloud_path = args.cloud_path
momentum = args.momentum

mobility = args.mobility
mobility_devices = args.mobility_devices
cosine = args.cosine
hierarchical = args.hierarchical
k2 = args.k2
sigma = args.sigma
ap_option = args.ap_option
aps = args.aps

print(f"Mobility: {mobility}, cosine: {cosine}, k2: {k2}, hierarchical: {hierarchical}")
if mobility == "true":
    mobility = True
else:
    mobility = False

if cosine == "true":
    cosine = True
else:
    cosine = False

if hierarchical == "true":
    hierarchical = True
else:
    hierarchical = False

config_dict = {}
if not path.exists("configs"):
    mkdir("configs")

with open(path.join("configs", cloud_configs_filename), 'w') as file:
    config_dict["experiment"] = cloud_configs_filename.split("_")[2]

    config_dict["cloud_path"] = cloud_path
    config_dict["cloud_ip"] = "127.0.0.1"
    config_dict["cloud_port"] = cloud_port
    config_dict["cloud_usr"] = cloud_usr
    config_dict["cloud_pwd"] = cloud_pwd
    config_dict["cloud_cuda_name"] = cloud_cuda

    config_dict["local_testing"] = local_testing
    config_dict["model_name"] = model_name

    config_dict["loss_func_name"] = "cross_entropy"
    config_dict["learning_rate"] = learning_rate
    config_dict["momentum"] = momentum

    config_dict["dataset_name"] = dataset_name
    config_dict["num_workers"] = 2
    config_dict["data_iid"] = data_iid
    config_dict["num_users"] = num_users
    config_dict["n_class"] = n_class
    config_dict["train_batch_size"] = train_batch_size
    config_dict["test_batch_size"] = 128

    config_dict["log_all"] = True
    config_dict["log_file"] = log_file
    config_dict["log_acc"] = True
    config_dict["log_loss"] = True
    config_dict["log_train_time"] = True
    config_dict["log_comm_time"] = True
    config_dict["log_power"] = True
    config_dict["log_temp"] = True
    config_dict["measurement_rate"] = 1

    config_dict["simulation"] = True
    config_dict["verbose"] = False

    config_dict["mobility"] = mobility
    config_dict["mobility_devices"] = mobility_devices
    config_dict["cosine"] = cosine
    config_dict["hierarchical"] = hierarchical
    config_dict["k2"] = k2
    config_dict["sigma"] = sigma
    config_dict["ap_option"] = ap_option
    config_dict["aps"] = aps

    json.dump(config_dict, file, indent=2)

# Device CONFIGS
dev_config_filename = args.dev_config_filename
num_devices = args.num_devices
dev_local_epochs = args.dev_local_epochs
dev_hw_types = args.dev_hw_types
ips = args.ips
ports = args.ports
cuda_names = args.cuda_names
train_batch_sizes = args.train_batch_sizes
model_names = args.model_names

aux = []
for k in ports[0].split(" "):
    aux.append(int(k))
ports = aux

aux = []
for k in train_batch_sizes[0].split(" "):
    aux.append(int(k))
train_batch_sizes = aux

aux = []
for k in ips[0].split(" "):
    aux.append(k)
ips = aux

aux = []
for k in dev_hw_types[0].split(" "):
    aux.append(k)
dev_hw_types = aux

aux = []
for k in cuda_names[0].split(" "):
    aux.append(k)
cuda_names = aux

aux = []
for k in dev_local_epochs[0].split(" "):
    aux.append(int(k))
dev_local_epochs = aux

aux = []
for k in model_names[0].split(" "):
    aux.append(k)
model_names = aux

config_dict = {}
with open(path.join("configs", dev_config_filename), 'w') as file:
    config_dict["num_devices"] = num_devices
    for dev_idx in range(num_devices):
        dev_dict = {}
        dev_dict["local_epochs"] = dev_local_epochs[dev_idx]
        dev_dict["hw_type"] = dev_hw_types[dev_idx]
        dev_dict["ip"] = ips[dev_idx]
        dev_dict["port"] = ports[dev_idx]
        dev_dict["train_batch_size"] = train_batch_sizes[dev_idx]
        dev_dict["cuda_name"] = cuda_names[dev_idx]
        dev_dict["model_name"] = model_names[dev_idx]
        config_dict[f"dev{dev_idx + 1}"] = dev_dict
    json.dump(config_dict, file, indent=2)

# Writing execution file sim.bash
if not path.exists("execs"):
    mkdir("execs")
with open(path.join("execs", exec_name), 'w') as file:
    file.write("#!/bin/bash\n\ndeclare -a elems=(\n")
    for dev_idx in range(num_devices):
        file.write(f"\t\"{ips[dev_idx]} {ports[dev_idx]}\"\n")
    file.write(")\n\nfor elem in \"${elems[@]}\"\ndo\n\tread -a tuple <<< \"$elem\"\n\tpython device.py ")
    file.write("--ip=\"${tuple[0]}\" --port=\"${tuple[1]}\" ")
    file.write(f"--exp={(cloud_configs_filename.split('_')[2]).split('p')[1]} --r=1 &")
    file.write(f"\ndone\n\n")
    file.write(f"python cloud.py --cloud_cfg {path.join('configs',cloud_configs_filename)} ")
    file.write(f"--dev_cfg {path.join('configs',dev_config_filename)} --seed {seed}")
