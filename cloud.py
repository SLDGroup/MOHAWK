import threading
from os import path
import os
import copy
import torch
import argparse
import json
import pickle
from models.get_model import get_model
import operator
from utils import connect, receive_msg, send_msg, close_connection, get_loss_func, get_hw_info, \
    receive_file, send_file, test, seed_everything
from fl_utils import aggregate_cos, aggregate_avg
import time

preferred_device = 'cpu'

class DeviceHandler(threading.Thread):
    def __init__(self, cloud_path, dev_idx, dev_ip, dev_port, dev_usr, dev_pwd, dev_path, dev_model_filename,
                 target_info, device_info, data, logs, local_testing, simulation, verbose, seed, end=False):
        threading.Thread.__init__(self)
        self.buffer_size = 1024
        self.recv_timeout = 10000

        self.connection = None

        self.cloud_path = cloud_path
        self.seed = seed
        seed_everything(self.seed)

        self.dev_model_filename = dev_model_filename
        self.target_info = target_info
        self.device_info = device_info
        self.data = data
        self.logs = logs
        self.local_testing = local_testing
        self.end = end

        self.dev_ip = dev_ip
        self.dev_port = dev_port
        self.dev_usr = dev_usr
        self.dev_pwd = dev_pwd
        self.dev_path = dev_path
        self.dev_idx = dev_idx
        self.dev_name = f"Device{self.dev_idx}"
        self.k_info_filename = f"dev_{self.dev_idx}_info.json"

        self.msg = None

        self.simulation = simulation
        self.verbose = verbose
        self.train_time = None

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self.train_time

    def set_comm_round(self, comm_round):
        self.msg = f"{self.target_info},{self.device_info},{comm_round};{self.data},{self.logs}"

    def run(self):
        # Step 1. Connect to server
        self.connection = connect(ip=self.dev_ip, port=self.dev_port, verbose=self.verbose)
        # Step 2. Send configs and wait for setup to be done
        if self.end:
            send_msg(connection=self.connection, msg=f"{self.msg},end", verbose=self.verbose)
            close_connection(connection=self.connection, verbose=self.verbose)
            exit()
        else:
            send_msg(connection=self.connection, msg=f"{self.msg},not_end", verbose=self.verbose)
        # Step 3. Synch with device with "done_setup"
        done_setup = receive_msg(connection=self.connection, buffer_size=self.buffer_size,
                                 recv_timeout=self.recv_timeout, verbose=self.verbose)
        assert done_setup is not "done_setup", f"[!] Received no input from {self.dev_name}"
        # Step 4. Send the device model weight
        if not self.simulation:
            send_file(connection=self.connection, target=self.dev_name,
                      target_ip=self.dev_ip, target_port=self.dev_port, target_usr=self.dev_usr,
                      target_pwd=self.dev_pwd, target_path=self.dev_path, filename=self.dev_model_filename,
                      dev_path=self.cloud_path, buffer_size=self.buffer_size, recv_timeout=self.recv_timeout,
                      verbose=self.verbose)
        # Step 5. Synch with the server
        received_data = receive_msg(connection=self.connection, buffer_size=self.buffer_size,
                                    recv_timeout=self.recv_timeout, verbose=self.verbose)
        if received_data.split(';')[0] != "done_training":
            print("[!] ERROR not done training")

        self.train_time = received_data.split(';')[1]

        # Step 6. Get the weights file from the server
        if not self.simulation:
            receive_file(connection=self.connection, target=self.dev_name, target_ip=self.dev_ip,
                         dev_path=self.cloud_path, buffer_size=self.buffer_size, recv_timeout=self.recv_timeout,
                         verbose=self.verbose, msg="local models zip")
        close_connection(connection=self.connection, verbose=self.verbose)


class Cloud:
    def __init__(self, cloud_cfg, dev_cfg, seed):
        self.cloud_cfg = cloud_cfg
        self.dev_cfg = dev_cfg
        self.seed = seed
        seed_everything(self.seed)

    def federated_learning(self):
        total_time_start = time.time()
        with open(self.cloud_cfg, 'r') as cfg:
            dat = json.load(cfg)

            experiment = dat["experiment"]

            cloud_path = dat["cloud_path"]
            cloud_ip = dat["cloud_ip"]
            cloud_port = dat["cloud_port"]
            cloud_usr = dat["cloud_usr"]
            cloud_pwd = dat["cloud_pwd"]
            cloud_cuda_name = dat["cloud_cuda_name"]

            local_testing = dat["local_testing"]
            model_name = dat["model_name"]

            loss_func_name = dat["loss_func_name"]
            learning_rate = dat["learning_rate"]
            momentum = dat["momentum"]

            dataset_name = dat["dataset_name"]
            num_workers = dat["num_workers"]
            data_iid = dat["data_iid"]
            num_users = dat["num_users"]
            train_batch_size = dat["train_batch_size"]
            test_batch_size = dat["test_batch_size"]

            log_all = dat["log_all"]
            log_file = dat["log_file"]
            log_acc = dat["log_acc"]
            log_loss = dat["log_loss"]
            log_train_time = dat["log_train_time"]
            log_comm_time = dat["log_comm_time"]
            log_power = dat["log_power"]
            log_temp = dat["log_temp"]

            simulation = dat["simulation"]
            verbose = dat["verbose"]

            mobility = dat["mobility"]
            mobility_devices = dat["mobility_devices"]
            cosine = dat["cosine"]
            hierarchical = dat["hierarchical"]
            k2 = dat["k2"]
            sigma = dat["sigma"]
            ap_option = dat["ap_option"]
            aps = dat["aps"]

            if aps > 0:
                aplist = []
                for ap_idx in range(0, aps):
                    aplist.append(f"AP{ap_idx}")

        if not os.path.exists(cloud_path):
            os.mkdir(cloud_path)
        if not os.path.exists("logs"):
            os.mkdir("logs")

        aux_model = get_model(model_name=f"{dataset_name}_{model_name}")
        net_glob = get_model(model_name=f"{dataset_name}_{model_name}")
        net_local = get_model(model_name=f"{dataset_name}_{model_name}")
        torch.save(net_glob.state_dict(), path.join(cloud_path, f"global_weights_0.pth"))
        loss_func = get_loss_func(loss_func_name=loss_func_name)

        loss_test, acc_test = test(model=net_glob, loss_func=loss_func, dataset_name=dataset_name,
                                   batch_size=test_batch_size, num_workers=num_workers,
                                   cuda_name=cloud_cuda_name, test_global=True, verbose=verbose, seed=self.seed, data_iid=data_iid, dev_idx=-1)

        print(f"Initial Accuracy: {acc_test}%; Initial Loss: {loss_test}")

        myaps = {}
        if hierarchical:
            """
            Get all APs to have same model (global model)
            """
            for ap_name in aplist:
                myaps[ap_name] = get_model(model_name=f"{dataset_name}_{model_name}")
                myweights = torch.load(path.join(cloud_path, f"global_weights_0.pth"), map_location=preferred_device)
                myaps[ap_name].load_state_dict(myweights, strict=False)
                torch.save(myaps[ap_name].to(torch.device(preferred_device)).state_dict(),
                           path.join(cloud_path, f"{ap_name}_weights_0.pth"))

        if not os.path.exists("logs"):
            os.mkdir("logs")

        if log_acc or log_loss:
            with open(path.join("logs", log_file), 'w') as logger:
                logger.write(f"CommRound,acc,loss\n")
        t = None
        t_idx = None
        mtype = ""
        if mobility:
            t_idx = 0
            prefix = "mobility_data/mobility_objects"
            if ap_option == "hierfavg" and cosine:
                mtype = "macfl"
            elif ap_option == "hierfavg" and not cosine:
                mtype = "hierfavg"
            elif ap_option == "use_only_trained_aps":
                mtype = "mohawk"
            else:
                print(f"ERROR: ap_option={ap_option}")
            print(f"{mtype} {mobility_devices}dev")
            with open(f"{prefix}/{mtype}_{mobility_devices}dev_{aps}ap_wifi_lte_100m_s{seed}.pkl", "rb") as f:
                print(f"Using: {mtype}_{mobility_devices}dev_{aps}ap_wifi_lte_100m_s{seed}.pkl")
                mobility_data_filt = pickle.load(f)

            t = list(mobility_data_filt.keys())
            myrange = range(len(t))
        else:
            t_idx = 0
            prefix = "mobility_data/mobility_objects"
            if ap_option == "hierfavg":
                mtype = "hierfavg"
            elif ap_option == "use_only_trained_aps":
                mtype = "mohawk"
            else:
                print(f"ERROR: ap_option={ap_option}")
            print(f"{mtype} {mobility_devices}dev")
            with open(f"{prefix}/{mtype}_{mobility_devices}dev_{aps}ap_wifi_lte_100m_s{seed}_nonmobility.pkl", "rb") as f:
                print(f"Using: {mtype}_{mobility_devices}dev_{aps}ap_wifi_lte_100m_s{seed}_nonmobility.pkl")
                mobility_data_filt = pickle.load(f)
            t = list(mobility_data_filt.keys())
            myrange = range(len(t))


        trained_until_now = []
        devices_last_seen = []
        if hierarchical:
            with open(f"logs/aps_{experiment}.csv", 'w') as logger:
                logger.write(f"Communication Round,Device Index,AP Name,Device Type,"
                             f"Communication Speedup, Distance to AP\n")
            if ap_option == "use_only_trained_aps":
                for idx in range(num_users):
                    devices_last_seen.append({"comm_round": -1, "aggregated": False})

        for comm_round in myrange:
            print(f"Mobility: {mobility}, cosine: {cosine}, hierarchical: {hierarchical}, "
                  f"total devices: {mobility_devices}, AP_OPTION: {ap_option}")
            if comm_round == 0:
                if log_acc or log_loss:
                    with open(path.join("logs", log_file), 'a+') as logger:
                        logger.write(f"{comm_round}")
                if log_acc:
                    with open(path.join("logs", log_file), 'a+') as logger:
                        logger.write(f",{acc_test}")
                if log_loss:
                    with open(path.join("logs", log_file), 'a+') as logger:
                        logger.write(f",{loss_test}")

            if log_acc or log_loss:
                with open(path.join("logs", log_file), 'a+') as logger:
                    logger.write(f"\n{comm_round + 1}")

            """
            Initialize the APs by loading global weights
            """
            if comm_round != 0:  # For communication rounds 1,...,N
                myweights = torch.load(path.join(cloud_path, f"global_weights_{comm_round - 1}.pth"),
                                       map_location=preferred_device)
                net_glob.load_state_dict(myweights, strict=False)
                if hierarchical:
                    for ap_name in aplist:
                        myweights = torch.load(path.join(cloud_path, f"{ap_name}_weights_{comm_round - 1}.pth"),
                                               map_location=preferred_device)
                        myaps[ap_name].load_state_dict(myweights, strict=False)
                if comm_round > 1:
                    if hierarchical:
                        for ap_name in aplist:
                            os.remove(path.join(cloud_path, f"{ap_name}_weights_{comm_round - 2}.pth"))
                    os.remove(path.join(cloud_path, f"global_weights_{comm_round - 2}.pth"))

            with open(self.dev_cfg, 'r') as f:
                dt = json.load(f)
                num_devices = dt["num_devices"]
                available_devices = []
                for d in mobility_data_filt[t[t_idx]]:
                    if d['internet_speed'] >= 0:
                        available_devices.append(d)
                # Skipping communication round - aggregate if necessary
                if len(available_devices) == 0:
                    print(f"[!] Skipping communication round {comm_round} for t={t[t_idx]}")
                    # if we skip, aggregate
                    if hierarchical:
                        if (comm_round + 1) % k2 == 0:
                            if ap_option == "hierfavg":
                                trained_until_now = aplist

                            ap_weights = []
                            print(f"Trained until now: {len(trained_until_now)}")
                            for ap_name in trained_until_now:
                                ap_weights.append(myaps[ap_name].to(torch.device(preferred_device)).state_dict())

                            if len(ap_weights) > 0:
                                if cosine:
                                    cloud_w = net_glob.to(torch.device(preferred_device)).state_dict()
                                    w_glob = aggregate_cos(sigma=sigma, global_weights=cloud_w,
                                                           local_weights=ap_weights)
                                elif not cosine:
                                    w_glob = aggregate_avg(local_weights=ap_weights)

                                for ap_name in aplist:
                                    torch.save(w_glob, path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                                torch.save(w_glob, path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                                print(f"Aggregated in {comm_round}: {trained_until_now}")
                                trained_until_now = []
                            else:
                                print(f"No global aggregation! - no APs trained")
                                for ap_name in aplist:
                                    torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                               path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                                torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                           path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                        else:  # If not global aggregation save APs and the global model
                            for ap_name in aplist:
                                torch.save(myaps[ap_name].to(torch.device(preferred_device)).state_dict(),
                                           path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                            torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                       path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                    else:  # If not hierarchical we do not have devices to aggregate
                        torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                   path.join(cloud_path, f"global_weights_{comm_round}.pth"))

                    loss_test, acc_test = test(model=net_glob, loss_func=loss_func, dataset_name=dataset_name,
                                               batch_size=test_batch_size, num_workers=num_workers,
                                               cuda_name=cloud_cuda_name, test_global=True, verbose=verbose,
                                               seed=self.seed, data_iid=data_iid, dev_idx=-1)
                    if log_acc:
                        with open(path.join("logs", log_file), 'a+') as logger:
                            logger.write(f",{acc_test}")
                    if log_loss:
                        with open(path.join("logs", log_file), 'a+') as logger:
                            logger.write(f",{loss_test}")
                    print(f"CommRound: {comm_round}; Accuracy: {acc_test}; Loss: {loss_test}")
                    t_idx += 1
                    continue
                elif len(available_devices) <= num_devices:
                    available_devices_idx = []
                    for d in available_devices:
                        available_devices_idx.append(d['device_idx'])
                elif len(available_devices) > 10:
                    available_devices_idx = []
                    for d in available_devices:
                        available_devices_idx.append(d['device_idx'])
                else:
                    print(f"[!] ERROR: for t={t[t_idx]}")
                print(f"Devices: {len(available_devices_idx)}")
                if hierarchical:
                    if ap_option == "use_only_trained_aps":
                        for dev_idx in available_devices_idx:
                            devices_last_seen[dev_idx] = {"comm_round": comm_round,
                                                          "aggregated": False}
                if simulation and mobility:
                    if ap_option == "hierfavg":
                        next_available_devices = []
                        next_available_devices_idx = []
                        for d in mobility_data_filt[t[t_idx+1]]:
                            if d['internet_speed'] >= 0:
                                if d['device_idx'] in available_devices_idx:
                                    next_available_devices.append(d)
                                    next_available_devices_idx.append(d['device_idx'])
                        if len(next_available_devices) > 0:
                            print(f"Training only {len(next_available_devices)} devices in comm round {comm_round}")
                            available_devices_idx = next_available_devices_idx
                            available_devices = next_available_devices
                    elif ap_option == "use_only_trained_aps":
                        next_available_devices = []
                        next_available_devices_idx = []
                        print(f"Time idx={t_idx}")
                        for i in range(1,k2+1):
                            if i > k2 - comm_round % k2 or t_idx + i > len(myrange):
                                print(len(myrange))
                                break
                            prev_len = len(next_available_devices_idx)
                            for d in mobility_data_filt[t[t_idx + i]]:
                                if d['internet_speed'] >= 0:
                                    if d['device_idx'] in available_devices_idx and d['device_idx'] not in next_available_devices_idx:
                                        next_available_devices.append(d)
                                        next_available_devices_idx.append(d['device_idx'])

                            print(f"\t Looking at future time idx={t_idx + i}; found {len(next_available_devices_idx)-prev_len} new devices")
                        if len(next_available_devices) > 0:
                            print(f"Training only {len(next_available_devices)} devices in comm round {comm_round}")
                            available_devices_idx = next_available_devices_idx
                            available_devices = next_available_devices

                # split available_devices_idx in series of <num_devices>
                mycnt = 0
                mydictcnt = 0
                avail_devs_idx_dict = {}
                for myidx in available_devices_idx:
                    if mycnt == 0:
                        avail_devs_idx_dict[mydictcnt] = [myidx]
                    else:
                        avail_devs_idx_dict[mydictcnt].append(myidx)
                    mycnt += 1
                    if mycnt == num_devices:
                        mycnt = 0
                        mydictcnt += 1
                for mydict_key in avail_devs_idx_dict.keys():
                    device_handler_list = []
                    for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                        if simulation:
                            dev = dt[f"dev{idx + 1}"]
                        else:
                            dev = dt[f"dev{i+1}"]
                        local_epoch = dev["local_epochs"]
                        dev_type = dev["hw_type"]
                        dev_ip = dev["ip"]
                        dev_port = dev["port"]
                        dev_cuda_name = dev["cuda_name"]
                        dev_train_batch_size = dev["train_batch_size"]
                        dev_model_name = dev["model_name"]

                        if hierarchical:
                            for d in available_devices:
                                if d['device_idx'] == i:
                                    dev_model_filename = f"dev_{i}.pth"
                                    if os.path.exists(path.join(cloud_path, dev_model_filename)):
                                        torch.save(myaps[d['AP_name'][0]].state_dict(),
                                                   path.join(cloud_path, dev_model_filename))
                                        with open(f"logs/aps_{experiment}.csv", 'a+') as logger:
                                            logger.write(f"{comm_round},{i},{d['AP_name'][0]},{d['device_type']},"
                                                         f"{d['internet_speed']},{d['dist_to_ap']}\n")
                                    else:
                                        torch.save(myaps[d['AP_name'][0]].state_dict(),
                                                   path.join(cloud_path, dev_model_filename))
                                        with open(f"logs/aps_{experiment}.csv", 'a+') as logger:
                                            logger.write(f"{comm_round},{i},{d['AP_name'][0]},{d['device_type']},"
                                                         f"{d['internet_speed']},{d['dist_to_ap']}\n")
                        elif not hierarchical:
                            for d in available_devices:
                                if d['device_idx'] == i:
                                    if not simulation:
                                        net_local = copy.deepcopy(net_glob).to(torch.device(dev_cuda_name))
                                    else:
                                        net_local = copy.deepcopy(net_glob)

                                    dev_model_filename = f"dev_{i}.pth"
                                    torch.save(net_local.state_dict(), path.join(cloud_path, dev_model_filename))

                        dev_pwd, dev_usr, dev_path = get_hw_info(dev_type)
                        """
                        {target_info},{device_info},{data},{logs},end
    
                        where
    
                        {target_info} = {experiment};{target_ip};{target_port};{target_path};{target_usr};{target_pwd}
                        {device_info} = {hw_type};{dev_idx};{data_iid};{num_users};{num_workers};{local_testing};
                                        {cuda_name};{simulation};{verbose};{seed}
                        {data}        = {comm_round}; # Comm round will be set in Device Handler
                                        {model_name};{filename};{local_epoch};{train_batch_size};
                                        {test_batch_size};{learning_rate};{momentum};{loss_func_name};{dataset_name}
                        {logs}        = {log_all};{log_file};{log_acc};{log_loss};{log_train_time};{log_comm_time};
                                        {log_power};{log_temp}
                        """
                        target_info = f"{experiment};{cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}"
                        device_info = f"{dev_type};{i};{data_iid};{num_users};" \
                                      f"{num_workers};{local_testing};{dev_cuda_name};{simulation};" \
                                      f"{verbose};{seed}"
                        data = f"{dev_model_name};{dev_model_filename};{local_epoch};" \
                               f"{dev_train_batch_size};{test_batch_size};{learning_rate};" \
                               f"{momentum};{loss_func_name};{dataset_name}"
                        logs = f"{log_all};{log_file};{log_acc};{log_loss};{log_train_time};{log_comm_time};{log_power};" \
                               f"{log_temp}"

                        device_handler_list.append(
                            DeviceHandler(cloud_path=cloud_path, dev_idx=i, dev_ip=dev_ip, dev_port=dev_port,
                                          dev_usr=dev_usr, dev_pwd=dev_pwd, dev_path=dev_path,
                                          dev_model_filename=dev_model_filename, target_info=target_info,
                                          device_info=device_info, data=data, logs=logs, local_testing=local_testing,
                                          simulation=simulation, verbose=verbose, seed=self.seed))

                    print(f"Started all clients - {len(avail_devs_idx_dict[mydict_key])}")
                    for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                        device_handler_list[idx].set_comm_round(comm_round=comm_round)
                        device_handler_list[idx].start()

                    print(f"\nWait until clients {avail_devs_idx_dict[mydict_key]} finish their job")
                    value = []
                    for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                        value.append(device_handler_list[idx].join())
                    print("Joined all clients")
            ############################################################################################################
            t_idx += 1  # Next timestep
            next_available_devices = []
            next_available_devices_idx = []
            for d in mobility_data_filt[t[t_idx]]:
                if ap_option == "hierfavg":
                    if d['internet_speed'] >= 0:
                        if d['device_idx'] in available_devices_idx:
                            next_available_devices.append(d)
                            next_available_devices_idx.append(d['device_idx'])
                elif ap_option != "hierfavg":
                    condition = devices_last_seen[d['device_idx']]["aggregated"] is False \
                                and devices_last_seen[d['device_idx']]["comm_round"] >= comm_round - (
                                            comm_round % k2)
                    if d['internet_speed'] >= 0 and condition:
                        next_available_devices.append(d)
                        next_available_devices_idx.append(d['device_idx'])

            print(f"{len(next_available_devices)} devices will aggregate in comm round {comm_round}")
            if len(next_available_devices) > 0:
                at_least_one = True
            else:
                at_least_one = False
            if not at_least_one:
                if hierarchical:
                    # Global aggregation
                    if (comm_round + 1) % k2 == 0:
                        print(f"[!] No edge aggregation this communication round!")
                        print(f"[+] Global aggregation this communication round!")
                        if ap_option == "hierfavg":
                            trained_until_now = aplist

                        ap_weights = []
                        print(f"Trained until now: {len(trained_until_now)}")
                        for ap_name in trained_until_now:
                            ap_weights.append(myaps[ap_name].to(torch.device(preferred_device)).state_dict())

                        if len(ap_weights) > 0:
                            if cosine:
                                cloud_w = net_glob.to(torch.device(preferred_device)).state_dict()
                                w_glob = aggregate_cos(sigma=sigma, global_weights=cloud_w,
                                                       local_weights=ap_weights)
                            elif not cosine:
                                w_glob = aggregate_avg(local_weights=ap_weights)

                            for ap_name in aplist:
                                torch.save(w_glob, path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                            torch.save(w_glob, path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                            print(f"Aggregated in {comm_round}: {trained_until_now}")
                            trained_until_now = []
                        else:
                            print(f"No global aggregation! - no APs trained")
                            for ap_name in aplist:
                                torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                           path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                            torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                       path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                    else:
                        for ap_name in aplist:
                            torch.save(myaps[ap_name].to(torch.device(preferred_device)).state_dict(),
                                       path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                        print(f"[!] No edge aggregation this communication round!")
                        torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                   path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                else:
                    print(f"[!] No global aggregation this communication round!")
                    torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                               path.join(cloud_path, f"global_weights_{comm_round}.pth"))
            elif at_least_one:
                if hierarchical:
                    # Device aggregation in the APs
                    local_weights = {}
                    for ap_name in aplist:
                        local_weights[ap_name] = []

                    for elem in next_available_devices:
                        if ap_option == "hierfavg":
                            condition = elem['device_idx'] in available_devices_idx
                        else:
                            condition = devices_last_seen[elem['device_idx']]["aggregated"] is False and \
                                        devices_last_seen[elem['device_idx']]["comm_round"] >= comm_round - (
                                                    comm_round % k2)
                        if condition:
                            if elem['AP_name'][0] not in trained_until_now and ap_option != "hierfavg":
                                trained_until_now.append(elem['AP_name'][0])

                            dev_model_filename = f"dev_{elem['device_idx']}.pth"
                            net_local = get_model(model_name=f"{dataset_name}_{model_name}")
                            net_local.load_state_dict(torch.load(path.join(cloud_path, dev_model_filename),
                                                                 map_location=preferred_device), strict=False)

                            w_local = net_local.to(torch.device(preferred_device)).state_dict()
                            if ap_option != "hierfavg":
                                local_weights[elem['AP_name'][0]].append(w_local)
                                devices_last_seen[elem['device_idx']]["aggregated"] = True
                            else:
                                local_weights[elem['AP_name'][0]].append(w_local)

                    for ap_name in aplist:
                        if len(local_weights[ap_name]) > 0:
                            if cosine:
                                w_ap = aggregate_cos(sigma=sigma, global_weights=myaps[ap_name].to(
                                    torch.device(preferred_device)).state_dict(), local_weights=local_weights[ap_name])
                            else:
                                w_ap = aggregate_avg(local_weights=local_weights[ap_name])

                            myaps[ap_name].load_state_dict(w_ap, strict=False)
                            torch.save(w_ap, path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                        else:
                            torch.save(myaps[ap_name].state_dict(), path.join(cloud_path,
                                                                              f"{ap_name}_weights_{comm_round}.pth"))

                    # Global aggregation
                    if (comm_round + 1) % k2 == 0:
                        if ap_option == "hierfavg":
                            trained_until_now = aplist

                        ap_weights = []
                        print(f"Trained until now: {len(trained_until_now)}")
                        for ap_name in trained_until_now:
                            ap_weights.append(myaps[ap_name].to(torch.device(preferred_device)).state_dict())

                        if len(ap_weights) > 0:
                            if cosine:
                                cloud_w = net_glob.to(torch.device(preferred_device)).state_dict()
                                w_glob = aggregate_cos(sigma=sigma, global_weights=cloud_w, local_weights=ap_weights)
                            elif not cosine:
                                w_glob = aggregate_avg(local_weights=ap_weights)

                            for ap_name in aplist:
                                torch.save(w_glob, path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                            torch.save(w_glob, path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                            print(f"Aggregated in {comm_round}: {trained_until_now}")
                            trained_until_now = []
                        else:
                            print(f"No global aggregation! - no APs trained")
                            for ap_name in aplist:
                                torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                           path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                            torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                       path.join(cloud_path, f"global_weights_{comm_round}.pth"))
                    else:
                        for ap_name in aplist:
                            torch.save(myaps[ap_name].to(torch.device(preferred_device)).state_dict(),
                                       path.join(cloud_path, f"{ap_name}_weights_{comm_round}.pth"))
                        torch.save(net_glob.to(torch.device(preferred_device)).state_dict(),
                                   path.join(cloud_path, f"global_weights_{comm_round}.pth"))

                elif not hierarchical:
                    local_weights = []
                    for elem in next_available_devices:
                        if elem['device_idx'] in available_devices_idx:
                            dev_model_filename = f"dev_{elem['device_idx']}.pth"

                            net_local = get_model(model_name=f"{dataset_name}_{model_name}")
                            net_local.load_state_dict(torch.load(path.join(cloud_path, dev_model_filename),
                                                                 map_location=preferred_device), strict=False)
                            w_local = net_local.state_dict()
                            local_weights.append(w_local)

                    cloud_w = net_glob.to(torch.device(preferred_device)).state_dict()
                    if cosine:
                        w_glob = aggregate_cos(sigma=sigma, global_weights=cloud_w, local_weights=local_weights)
                    else:
                        w_glob = aggregate_avg(local_weights=local_weights)
                    torch.save(w_glob, path.join(cloud_path, f"global_weights_{comm_round}.pth"))

            net_glob.load_state_dict(torch.load(path.join(cloud_path, f"global_weights_{comm_round}.pth")),
                                     strict=False)
            loss_test, acc_test = test(model=net_glob, loss_func=loss_func, dataset_name=dataset_name,
                                       batch_size=test_batch_size, num_workers=num_workers, cuda_name=cloud_cuda_name,
                                       test_global=True, verbose=verbose, seed=self.seed, data_iid=data_iid, dev_idx=-1)

            if log_acc:
                with open(path.join("logs", log_file), 'a+') as logger:
                    logger.write(f",{acc_test}")
            if log_loss:
                with open(path.join("logs", log_file), 'a+') as logger:
                    logger.write(f",{loss_test}")
            print(f"CommRound: {comm_round}; Accuracy: {acc_test}; Loss: {loss_test}")
        print(f"Total time for experiment: {time.time() - total_time_start}")
        with open(path.join("logs", "time.csv"), 'a+') as logger:
            logger.write(f"{experiment},{time.time() - total_time_start}\n")
        print("Closing everything")
        device_handler_list = []
        with open(self.dev_cfg, 'r') as f:
            dt = json.load(f)
            num_devices = dt["num_devices"]
            for i in range(num_devices):
                dev = dt[f"dev{i + 1}"]
                local_epoch = dev["local_epochs"]
                dev_type = dev["hw_type"]
                dev_ip = dev["ip"]
                dev_port = dev["port"]
                dev_cuda_name = dev["cuda_name"]

                net_local = net_glob.to(torch.device(dev_cuda_name))

                dev_model_filename = f"dev_{i}.pth"
                torch.save(net_local.state_dict(), path.join(cloud_path, dev_model_filename))

                dev_pwd, dev_usr, dev_path = get_hw_info(dev_type)
                if dev_type == 'lambda':
                    dev_path = cloud_path
                """
                {target_info},{device_info},{data},{logs},end

                where

                {target_info} = {experiment};{target_ip};{target_port};{target_path};{target_usr};{target_pwd}
                {device_info} = {hw_type};{dev_idx};{data_iid};{num_users};
                                {num_workers};{local_testing};{cuda_name};{simulation};
                                {verbose};{seed}
                {data}        = {comm_round}; # Comm round will be set in Device Handler
                                {model_name};{filename};{local_epoch};{train_batch_size};
                                {test_batch_size};{learning_rate};{momentum};{loss_func_name};{dataset_name}
                {logs}        = {log_all};{log_file};{log_acc};{log_loss};{log_train_time};{log_comm_time};
                                {log_power};{log_temp}
                """
                target_info = f"{experiment};{cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}"
                device_info = f"{dev_type};{i};{data_iid};{num_users};" \
                              f"{num_workers};{local_testing};{dev_cuda_name};{simulation};" \
                              f"{verbose};{seed}"
                data = f"{model_name};{dev_model_filename};{local_epoch};{train_batch_size};" \
                       f"{test_batch_size};{learning_rate};{momentum};{loss_func_name};{dataset_name}"
                logs = f"{log_all};{log_file};{log_acc};{log_loss};{log_train_time};{log_comm_time};{log_power};" \
                       f"{log_temp}"

                device_handler_list.append(
                    DeviceHandler(cloud_path=cloud_path, dev_idx=i, dev_ip=dev_ip, dev_port=dev_port,
                                  dev_usr=dev_usr, dev_pwd=dev_pwd, dev_path=dev_path,
                                  dev_model_filename=dev_model_filename, target_info=target_info,
                                  device_info=device_info, data=data, logs=logs, local_testing=local_testing,
                                  simulation=simulation, verbose=verbose, seed=self.seed, end=True))

        print("Closing all clients")
        for i in range(num_devices):
            device_handler_list[i].set_comm_round(comm_round=-1)
            device_handler_list[i].start()

        print("\nWait until clients close")
        for i in range(num_devices):
            device_handler_list[i].join()
        print("Closed all clients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_cfg', type=str, default="configs/cloud_cfg_exp1.json",
                        help='Cloud configuration file name')
    parser.add_argument('--dev_cfg', type=str, default="configs/dev_cfg_exp1.json",
                        help='Device configuration file name')
    parser.add_argument('--seed', type=int, default=42, help='Seed for experiments')
    args = parser.parse_args()

    cloud_cfg = args.cloud_cfg
    dev_cfg = args.dev_cfg
    seed = args.seed

    cloud = Cloud(cloud_cfg=cloud_cfg, dev_cfg=dev_cfg, seed=seed)
    cloud.federated_learning()
