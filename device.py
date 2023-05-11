from os import path
import os
import socket
import torch
import argparse
import threading
import time
import telnetlib
from sys import getsizeof
from utils import receive_msg, send_msg, close_connection, local_training, get_loss_func, get_hw_info, receive_file, \
    send_file, zip_file, seed_everything

class Device:
    def __init__(self, connection, dev_ip_port, exp, r, dev_idx):
        """
        Initializes the client. For each device we need one client to be initialized.

        Parameters
        ----------
        connection : socket.socket
            Socket connection with the Server.

        dev_ip_port : tuple
            The client IP and port. (hostaddr, port)
        """

        self.connection = connection
        self.dev_ip = dev_ip_port[0]
        self.dev_port = int(dev_ip_port[1])

        self.seed = None
        self.dev_pwd, self.dev_usr, self.dev_path = None, None, None

        self.buffer_size = 1024
        self.recv_timeout = 10000
        self.experiment = ""
        self.dataset_train, self.dict_users_train, self.dataset_test, self.dict_users_test = None, None, None, None

        self.target, self.target_ip, self.target_port, self.target_path, self.target_usr, self.target_pwd = \
            None, None, None, None, None, None

        self.hw_type, self.dev_idx, self.data_iid = None, None, None
        self.num_users = None, None
        self.local_testing, self.cuda_name, self.simulation, self.verbose = None, None, None, None
        self.num_workers = None
        self.model_filename, self.learning_rate = None, None

        self.model_name, self.comm_round = None, None
        self.local_epochs, self.train_batch_size, self.test_batch_size = None, None, None
        self.learning_rate, self.momentum = None, None
        self.loss_func, self.loss_func_name, self.model_path, self.dataset_name = None, None, None, None

        self.logs, self.log_all, self.log_file = None, None, None
        self.log_acc, self.log_loss, self.log_train_time, self.log_comm_time = True, True, True, True
        self.log_power, self.log_temp = True, True

        self.ksize, self.filename_shift, self.server_idx = None, None, None
        self.average, self.num_batches_tracked_avg = None, None
        self.device = None
        self.dict_users_trn, self.dict_users_tst = None, None

        self.exp = exp
        self.r = r
        self.dev_idx = dev_idx
        self.log_file = f"log_exp{self.exp}_run{self.r}.csv"

    def run(self):
        """
        Main method for the Device. Cloud sends a message to the Device in the following format:

            {target_info},{device_info},{data},{logs}

            where

            {target_info} = {experiment};{target_ip};{target_port};{target_path};{target_usr};{target_pwd}
            {device_info} = {hw_type};{dev_idx};{data_iid};{num_users};
                            {num_workers};{local_testing};{cuda_name};{simulation};
                            {verbose};{seed}
            {data}        = {comm_round};{model_name};{filename};{local_epochs};{train_batch_size};
                            {test_batch_size};{learning_rate};{momentum};{loss_func_name};{dataset_name}
            {logs}        = {log_all};{log_file};{log_acc};{log_loss};{log_train_time};{log_comm_time};{log_power};
                            {log_temp}
        """

        msg = receive_msg(self.connection, self.buffer_size, self.recv_timeout, verbose=self.verbose)
        if self.log_comm_time:
            filesize = getsizeof(msg)
        assert msg is not None, "[!] Received no input from client"

        target_info, device_info, data, logs, end = msg.split(',')

        self.experiment, self.target_ip, self.target_port, self.target_path, self.target_usr, self.target_pwd = target_info.split(';')
        self.target = "Cloud"

        self.hw_type, self.dev_idx, self.data_iid, self.num_users, \
            self.num_workers, self.local_testing, self.cuda_name, \
            self.simulation, self.verbose, self.seed = device_info.split(';')
        self.seed = int(self.seed)
        seed_everything(self.seed)

        self.dev_pwd, self.dev_usr, self.dev_path = get_hw_info(self.hw_type)
        self.dev_idx = int(self.dev_idx)
        self.data_iid = bool(int(self.data_iid == 'True'))
        self.num_users = int(self.num_users)
        self.num_workers = int(self.num_workers)
        self.local_testing = bool(int(self.local_testing == 'True'))
        self.simulation = bool(int(self.simulation == 'True'))
        self.verbose = bool(int(self.verbose == 'True'))

        self.comm_round, self.model_name, self.model_filename, self.local_epochs, self.train_batch_size, \
            self.test_batch_size, self.learning_rate, self.momentum, \
            self.loss_func_name, self.dataset_name = data.split(';')

        self.comm_round = int(self.comm_round)
        self.local_epochs = int(self.local_epochs)
        self.train_batch_size = int(self.train_batch_size)
        self.test_batch_size = int(self.test_batch_size)
        self.learning_rate = float(self.learning_rate)
        self.momentum = float(self.learning_rate)
        self.loss_func = get_loss_func(self.loss_func_name)
        self.model_path = path.join(self.dev_path, self.model_filename)

        logs = logs.split(';')
        if self.simulation:
            self.log_all  = False
            self.log_file = logs[1]
            self.log_acc  = False
            self.log_loss = False
            self.log_train_time = bool(int(logs[4] == 'True'))
            self.log_comm_time = True
            self.log_power = False
            self.log_temp  = False
        else:
            self.log_all = bool(int(logs[0] == 'True'))
            self.log_file = logs[1]
            if not self.log_all:
                self.log_acc = bool(int(logs[2] == 'True'))
                self.log_loss = bool(int(logs[3] == 'True'))
                self.log_train_time = bool(int(logs[4] == 'True'))
                self.log_comm_time = bool(int(logs[5] == 'True'))
                self.log_power = bool(int(logs[6] == 'True'))
                self.log_temp = bool(int(logs[7] == 'True'))

        self.device = torch.device(self.cuda_name)

        if not os.path.exists(self.dev_path):
            os.mkdir(self.dev_path)


        if end == "end":
            close_connection(connection=self.connection, verbose=self.verbose)
            return True

        # Step 2. Synch with Cloud with msg "done_setup"
        if self.log_comm_time:
            comm_time = 0.
            start_time = time.time()
        send_msg(connection=self.connection, msg="done_setup", verbose=self.verbose)
        if self.log_comm_time:
            filesize += getsizeof("done_setup")
            comm_time += time.time() - start_time

        # print("Step 3. Receive local model weight file")
        if not self.simulation:
            if self.log_comm_time:
                start_time = time.time()
            filesize += receive_file(connection=self.connection, target=self.target, target_ip=self.target_ip,
                                     dev_path=self.dev_path, buffer_size=self.buffer_size,
                                     recv_timeout=self.recv_timeout, verbose=self.verbose, msg="model weight zip")
            if self.log_comm_time:
                comm_time += time.time() - start_time
        else:
            _, myf = zip_file(filename=self.model_path, target_path=".", verbose=False)
            filesize += myf
            if self.log_comm_time:
                comm_time += time.time() - start_time

        # Step 4. Train the model
        train_loss, train_acc, test_loss, test_acc, train_time = \
            local_training(model_name=self.model_name, dataset_name=self.dataset_name, loss_func=self.loss_func,
                           batch_size=self.train_batch_size, num_workers=self.num_workers, model_path=self.model_path,
                           local_testing=self.local_testing, cuda_name=self.cuda_name, learning_rate=self.learning_rate,
                           momentum=self.momentum, local_epochs=self.local_epochs, dev_idx=self.dev_idx,
                           log_train_time=self.log_train_time, verbose=self.verbose, data_iid=self.data_iid,
                           seed=self.seed)
        before_train_filesize = filesize
        before_train_comm_time = comm_time
        filesize = 0
        comm_time = 0.
        # Step 5. Synch by sending to Cloud "done_training"
        if self.log_comm_time:
            start_time = time.time()
        send_msg(connection=self.connection, msg=f"done_training;{train_time}", verbose=self.verbose)
        filesize += getsizeof(f"done_training;{train_time}")
        if self.log_comm_time:
            comm_time += time.time() - start_time

        # Step 6. Send the model weights  back
        if not self.simulation:
            if self.log_comm_time:
                start_time = time.time()
            filesize += send_file(connection=self.connection, target=self.target, target_ip=self.target_ip,
                                  target_port=self.target_port, target_usr=self.target_usr, target_pwd=self.target_pwd,
                                  target_path=self.target_path, filename=self.model_filename, dev_path=self.dev_path,
                                  buffer_size=self.buffer_size, recv_timeout=self.recv_timeout, verbose=self.verbose)
            if self.log_comm_time:
                comm_time += time.time() - start_time
        else:
            _, myf = zip_file(filename=self.model_path, target_path=".", verbose=False)
            filesize += myf
            if self.log_comm_time:
                comm_time += time.time() - start_time
        after_train_filesize = filesize
        after_train_comm_time = comm_time
        close_connection(connection=self.connection, verbose=self.verbose)
        return False


def run(stop, device_type, measurement_rate, dev_idx, log_power, log_temp, exp, r):
    if device_type != 'nano':
        if device_type == 'rpi':
            import gpiozero
        elif device_type == 'mc1':
            import sysfs_paths

        if not os.path.exists(f"logs"):
            os.mkdir(f"logs")

        if log_power:
            sp2 = telnetlib.Telnet('192.168.4.1')
            time.sleep(2)
            pwr = 0

        if log_power and log_temp:
            print(f"[+] Recording power and temperature...")
            with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'w') as logger:
                logger.write("Timestamp,Power,Temperature\n")
        elif log_power and not log_temp:
            print(f"[+] Recording power ...")
            with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'w') as logger:
                logger.write("Timestamp,Power\n")
        elif not log_power and log_temp:
            print(f"[+] Recording temperature...")
            with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'w') as logger:
                logger.write("Timestamp,Temperature\n")

        last_power = 0.0
        while True:
            loop_start = time.time()

            if log_power:
                sp2_readings = str(sp2.read_eager())
                # Find latest power measurement in the data
                idx = sp2_readings.rfind('\n')
                idx2 = sp2_readings[:idx].rfind('\n')
                idx2 = idx2 if idx2 != -1 else 0
                ln = sp2_readings[idx2:idx].strip().split(',')
                if len(ln) < 2:
                    pwr = last_power
                else:
                    pwr = float(ln[-2])
                    last_power = pwr

            if log_temp:
                temp = 0
                if device_type == 'mc1':
                    for i in range(4):
                        with open(sysfs_paths.fn_thermal_sensor.format(i)) as f:
                            temp += int(f.read()) / 1000
                    temp /= 4
                elif device_type == 'rpi':
                    temp = gpiozero.CPUTemperature().temperature

            if log_power and log_temp:
                with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'a+') as logger:
                    logger.write(f"{time.time()},{pwr},{temp}\n")
            elif log_power and not log_temp:
                with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'a+') as logger:
                    logger.write(f"{time.time()},{pwr}\n")
            elif not log_power and log_temp:
                with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'a+') as logger:
                    logger.write(f"{time.time()},{temp}\n")

            time.sleep(max(0., measurement_rate - (time.time() - loop_start)))
            if stop():
                break
    else:
        if not os.path.exists(f"logs"):
            os.mkdir(f"logs")
        os.system("echo password | sudo -S tegrastats --stop")
        os.system("echo password | sudo -S tegrastats --interval 1000 --logfile tegrastats.log &")
        time.sleep(2)
        print(f"[+] Recording power and temperature...")
        file = "tegrastats.log"
        with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'w') as logger:
            logger.write(f"Timestamp,RAM,SWAP,CPU1 %,CPU1_FREQ MHz,CPU2 %,CPU2_FREQ MHz,CPU3 %,CPU3_FREQ MHz,CPU4 %,CPU4_FREQ MHz,CPU_TEMP C,GPU %,GPU_FREQ MHz,GPU_TEMP C,TOTAL_TEMP,TOTAL_POWER mW,TOTAL_AVG mW,GPU_POW mW,GPU_AVG_POW mW,CPU_POW mW,CPU_AVG_POW mW\n")
        while True:
            loop_start = time.time()
            with open(file, "rb") as f:
                first = f.readline()  # Read the first line.
                f.seek(-2, os.SEEK_END)  # Jump to the second last byte.
                while f.read(1) != b"\n":  # Until EOL is found...
                    f.seek(-2, os.SEEK_CUR)  # ...jump back the read byte plus one more.
                last = f.readline()  # Read last line.
            with open(path.join("logs", f"device{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv"), 'a+') as logger:
                ram = (str(last).split("RAM "))[1].split("/")[0]
                swap = (str(last).split("SWAP "))[1].split("/")[0]
                cpus = (str(last).split("CPU ["))[1].split("] EMC_FREQ")[0].split(",")

                cpu_val = []
                cpu_freq = []
                for cpu in cpus:
                    cpu_val.append(cpu.split("%")[0])
                    cpu_freq.append(cpu.split("@")[1])

                cpu_temp = (str(last).split("CPU@"))[1].split("C PMIC@")[0]
                gpu = (str(last).split("GR3D_FREQ "))[1].split("%")[0]
                gpu_freq = ((str(last).split("GR3D_FREQ "))[1].split(" VIC_FREQ")[0]).split("@")[1]
                gpu_temp = (str(last).split("GPU@"))[1].split("C AO@")[0]
                thermal = (str(last).split("thermal@"))[1].split("C POM_5V")[0]

                power = ((str(last).split("POM_5V_IN "))[1].split(" POM_5V_GPU")[0]).split("/")[0]
                avg = ((str(last).split("POM_5V_IN "))[1].split(" POM_5V_GPU")[0]).split("/")[1]
                gpu_power = ((str(last).split("POM_5V_GPU "))[1].split(" POM_5V_CPU")[0]).split("/")[0]
                gpu_avg = ((str(last).split("POM_5V_GPU "))[1].split(" POM_5V_CPU")[0]).split("/")[1]
                cpu_power = (str(last).split("POM_5V_CPU "))[1].split("/")[0]
                cpu_avg = ((str(last).split("POM_5V_CPU "))[1].split('\\')[0]).split("/")[1]
                logger.write(f"{time.time()},{ram},{swap},{cpu_val[0]},{cpu_freq[0]},{cpu_val[1]},{cpu_freq[1]},"
                             f"{cpu_val[2]},{cpu_freq[2]},{cpu_val[3]},{cpu_freq[3]},{cpu_temp},{gpu},{gpu_freq},"
                             f"{gpu_temp},{thermal},{power},{avg},{gpu_power},{gpu_avg},{cpu_power},{cpu_avg}\n")
                # RAM 957/3964MB (lfb 528x4MB) SWAP 0/1982MB (cached 0MB) IRAM 0/252kB(lfb 252kB)
                # CPU [11%@102,1%@102,1%@204,1%@204] EMC_FREQ 1%@204 GR3D_FREQ 0%@76 VIC_FREQ 0%@192 APE 25 PLL@21C
                # CPU@23.5C PMIC@100C GPU@22C AO@32C thermal@22.5C POM_5V_IN 998/987 POM_5V_GPU 41/30
                # POM_5V_CPU 124/113\n
                time.sleep(max(0., measurement_rate - (time.time() - loop_start)))
            if stop():
                os.system("echo password | sudo -S tegrastats --stop")
                break


def start_device(host, port, device_type, measurement_rate, dev_idx, log_power, log_temp, exp, r):
    soc = socket.socket()
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    soc.bind((host, port))
    if log_power or log_temp:
        stop_threads = False
        t1 = threading.Thread(target=run, args=(lambda: stop_threads, device_type, measurement_rate, dev_idx,
                                                log_power, log_temp, exp, r))
        t1.start()
    while True:
        try:
            soc.listen(5)
            connection, dev_ip_port = soc.accept()
            server = Device(connection=connection, dev_ip_port=dev_ip_port, exp=exp, r=r, dev_idx=dev_idx)
            br = server.run()
            if br:
                if log_power or log_temp:
                    stop_threads = True
                    t1.join()
                break
        except BaseException as e:
            soc.close()
            print(f"[!] TIMEOUT Socket Closed Because no Connections Received.\n[!] ERROR: {e}\n")
            if log_power or log_temp:
                stop_threads = True
                t1.join()
            break


parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, default="127.0.0.1", help='IP address')
parser.add_argument('--port', type=int, default=10001, help='Port')
parser.add_argument('--device_type', type=str, default='rpi', help='Device hw type')
parser.add_argument('--measurement_rate', type=float, default=1, help='Measurement rate in seconds')
parser.add_argument('--log_file', type=str, default="log_file.csv", help='Log file name')
parser.add_argument('--log_power', type=bool, default=False, help='Log power?')
parser.add_argument('--log_temp', type=bool, default=False, help='Log temperature?')
parser.add_argument('--dev_idx', type=int, default=0, help='Device number')
parser.add_argument('--exp', type=int, default=0,  help='Device number')
parser.add_argument('--r', type=int, default=0, help='Device number')
args = parser.parse_args()

start_device(host=args.ip, port=args.port, device_type=args.device_type, measurement_rate=args.measurement_rate,
             dev_idx=args.dev_idx, log_power=args.log_power, log_temp=args.log_temp, exp=args.exp, r=args.r)
