import pickle
import time
from os import path
import os
import socket
import zipfile
import paramiko
from scp import SCPClient
import sys
from torchvision import datasets, transforms
from os.path import basename
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.get_model import get_model
import _pickle as cPickle
from tqdm import tqdm

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def connect(ip, port, verbose):
    """
    Function for making a connection with the Device.
    """
    # Creating a socket
    soc = socket.socket()
    if verbose:
        print("[+] Socket is created.")

    # Connecting to the Client
    if verbose:
        print(f"[+] Trying to connect to the server {ip} at port {port}")
    retry = True
    while retry:
        try:
            soc.connect((ip, port))
            retry = False
        except BaseException as e:
            print(f"[!] ERROR: {e}")
            retry = True
            print("Retrying to connect...")
            time.sleep(5)

    if verbose:
        print(f"[+] Connected to the server {ip} at port {port}\n")
    return soc


def receive_file(connection, target, target_ip, dev_path, buffer_size, recv_timeout, verbose, msg="zip"):
    """
    Function for receiving the global model from the Cloud Server. Until a successful file transfer, the
    Device tries again to get the global model weight file.
    """

    if verbose:
        print(f"[+] Transferring {msg} from the {target}")

    # Step 1. Get the filename and filesize that were transmitted
    filename, filesize = get_notification_transfer_done(connection, buffer_size,
                                                        recv_timeout, target, verbose)
    zipped_file = filename.split(".")[0] + ".zip"
    while True:
        # Step 2. Receive the file and unzip it
        unzip_file(connection=connection, zip_filename=zipped_file,
                   target_path=dev_path, verbose=verbose)
        # Step 3. Check file size. If ok send "Confirm" message, otherwise send "Resend" message
        # until successful transmission
        if path.exists(path.join(dev_path, zipped_file)):
            if os.path.getsize(path.join(dev_path, zipped_file)) == int(filesize):
                if verbose:
                    print(f"[+] Confirming successful reception of the file to {target_ip}")
                send_msg(connection, "Confirm", verbose)
                break
            else:
                if verbose:
                    print(f"[-] File {zipped_file} size not ok with "
                          f"{os.path.getsize(path.join(dev_path, zipped_file))} != {filesize}")
                send_msg(connection, "Resend", verbose)
                filename, filesize = get_notification_transfer_done(connection, buffer_size, recv_timeout,
                                                                    target, verbose)
        else:
            if verbose:
                print(f"[-] File {zipped_file} path not ok in {dev_path}")
            send_msg(connection, "Resend", verbose)
            filename, filesize = get_notification_transfer_done(connection, buffer_size, recv_timeout, target, verbose)
    if verbose:
        print(f"[+] Transfer of {msg} done successfully!")
    return filesize


def send_file(connection, target, target_ip, target_port, target_usr, target_pwd, target_path, filename, dev_path,
              buffer_size, recv_timeout, verbose):
    """
    Function for sending the local model for the Cloud Server.
    After training the local model, the Device sends the message "ready" to the Cloud Server. Then,
    the Device transfers the weight file. It sends another message to the Client in the following format:

        {data}

        where

        {data} = {filename};{filesize}

    Then, the Server waits for a message from the Client. If "Confirm" is received, the connection is closed. If
    "Resend" is received, the file is resent until a "Confirm" message is received from the Client.
    """

    if verbose:
        print(f"[+] Sending local model to the {target}")

    zip_filename, filesize = zip_file(filename=filename, target_path=dev_path, verbose=verbose)

    if verbose:
        print(f"[+] Sending the {zip_filename} to {target_ip}")

    scp_file(target_ip=target_ip, target_port=target_port, target_usr=target_usr, target_pwd=target_pwd,
             target_path=target_path, zip_filename=zip_filename, source_path=dev_path, verbose=verbose)

    if verbose:
        print("[+] Weights sent\n")
        print(f"[+] Sending confirmation of transmission to the {target}.")

    send_msg(connection, f"{filename};{filesize}", verbose)
    if verbose:
        print(f"[+] Getting confirmation from the {target}")
    received_data = receive_msg(connection, buffer_size, recv_timeout, verbose)

    while received_data != "Confirm":
        if received_data == "Resend":
            if verbose:
                print(f"[+] RESENDING the {zip_filename} to {target_ip}")
            scp_file(target_ip=target_ip, target_port=target_port, target_usr=target_usr, target_pwd=target_pwd,
                     target_path=target_path, zip_filename=zip_filename, source_path=dev_path, verbose=verbose)

            if verbose:
                print("[+] Weights sent\n")
                print(f"[+] Sending confirmation of transmission to the {target}.")
            send_msg(connection, f"{filename};{filesize}", verbose)
            if verbose:
                print(f"[+] Getting confirmation from the {target}")
            received_data = receive_msg(connection, buffer_size, recv_timeout, verbose)
    if verbose:
        print(f"[+] File sent successfully to {target} {target_ip}")
    return filesize


def get_hw_info(hw_type):
    if hw_type == "rpi":
        return "password", "pi", "/home/pi/MOHAWK/files/"
    elif hw_type == "mc1":
        return "password", "odroid", "/home/odroid/MOHAWK/files/"
    elif hw_type == "local":
        return "pwd", "usr", "/home/usr/MOHAWK/files/"
    else:
        print("[!] ERROR wrong device type.")


def get_loss_func(loss_func_name):
    if loss_func_name == "cross_entropy":
        return nn.CrossEntropyLoss()


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        for i in range(len(self.idxs)):
            self.idxs[i] = int(self.idxs[i])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



def train(model, loss_func, dev_idx, batch_size, num_workers, model_path, cuda_name, optimizer, local_epochs,
          verbose, dataset_name, seed, data_iid):

    device = torch.device(cuda_name)
    if data_iid:
        iidtype = "iid"
    else:
        iidtype = "niid"
    with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/imgs_train_dev{dev_idx}.pkl", 'rb') as f:
        imgs_train = cPickle.load(f)
    with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/labels_train_dev{dev_idx}.pkl", 'rb') as f:
        labels_train = cPickle.load(f)
    with open(f"dataset/{dataset_name}/iid/seed{seed}/transform_train.pkl", 'rb') as f:
        transform_train = cPickle.load(f)
    data_loader = DataLoader(DatasetSplitDirichlet(image=imgs_train, target=labels_train, transform=transform_train),
                             batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model.train()

    for epoch in range(local_epochs):
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 1
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, act = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        if verbose:
            print(f"Epoch {epoch} loss: {train_loss/(batch_idx+1)} accuracy: {100.*correct/total}")

    torch.save(model.state_dict(), model_path)
    return train_loss/(batch_idx+1), (100.*correct/total)


def test(model, loss_func, dev_idx, batch_size, num_workers, cuda_name, test_global, verbose,
         dataset_name, seed, data_iid, return_total=False):

    device = torch.device(cuda_name)
    model = model.to(device)

    if test_global:
        with open(f"dataset/{dataset_name}/iid/seed{seed}/global_test.pkl", 'rb') as f:
            dataset_test = cPickle.load(f)
        data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        if data_iid:
            iidtype = "iid"
        else:
            iidtype = "niid"
        with open(f"dataset/{dataset_name}/iid/seed{seed}/transform_test.pkl", 'rb') as f:
            transform_test = cPickle.load(f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/imgs_test_dev{dev_idx}.pkl", 'rb') as f:
            imgs_test = cPickle.load(f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/labels_test_dev{dev_idx}.pkl", 'rb') as f:
            labels_test = cPickle.load(f)
        data_loader = DataLoader(DatasetSplitDirichlet(image=imgs_test, target=labels_test, transform=transform_test),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 1

    if not test_global and verbose:
        print(f"[++] Testing on local dataset... ")
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, act = model(images)
            loss = loss_func(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if verbose:
            print(f'[++] Local model loss {test_loss/(batch_idx+1)}, Local model accuracy: {100.*correct/total}%')
    if not test_global and verbose:
        print(f"[++] Testing on local dataset... ")
        print(f"[++] Finished testing in {time.time() - start_time}")

    if return_total:
        return test_loss/(batch_idx+1), 100.*correct/total, total
    else:
        return test_loss/(batch_idx+1), 100.*correct/total


def local_training(model_name, dataset_name, loss_func, batch_size, num_workers, model_path, local_testing, cuda_name,
                   learning_rate, momentum, local_epochs, log_train_time, dev_idx, verbose, data_iid, seed):

    device = torch.device(cuda_name)
    model = get_model(model_name=f"{dataset_name}_{model_name}")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    if verbose:
        print(f"[++] Device{dev_idx} training...")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if log_train_time:
        start_time = time.time()
    train_loss, train_acc = train(model=model, loss_func=loss_func, batch_size=batch_size, num_workers=num_workers,
                                  model_path=model_path, cuda_name=cuda_name, optimizer=optimizer,
                                  local_epochs=local_epochs, verbose=verbose, dataset_name=dataset_name, seed=seed,
                                  data_iid=data_iid, dev_idx=dev_idx)
    if log_train_time:
        train_time = time.time() - start_time
    if verbose:
        print(f"[++] Train loss: {train_loss}")

    test_loss, test_acc = None, None
    if local_testing:
        if verbose:
            print("[++] Evaluating local accuracy after training...")
        test_loss, test_acc = test(model=model, loss_func=loss_func, batch_size=batch_size, num_workers=num_workers,
                                   cuda_name=cuda_name, test_global=False, verbose=verbose, dataset_name=dataset_name,
                                   seed=seed, data_iid=data_iid, dev_idx=dev_idx)
        print(f"Train loss {train_loss} Test loss {test_loss} Train acc {train_acc} Test acc {test_acc}")
    return train_loss, train_acc, test_loss, test_acc, train_time


class DatasetSplitDirichlet(Dataset):
    def __init__(self, image, target, transform):
        self.image = image
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        image = image / 255.
        transform = transforms.ToPILImage()
        image = transform(image)
        image = self.transform(image)
        target = self.target[index]
        return image, target


def dirichlet(dataset, num_users, images_per_client, alpha, dataset_name):
    num_classes = len(dataset.classes)
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        idx = [torch.where(torch.FloatTensor(dataset.targets) == i) for i in range(num_classes)]
        data = [dataset.data[idx[i][0]] for i in range(num_classes)]
    else:
        idx = [torch.where(dataset.targets == i) for i in range(num_classes)]
        data = [dataset.data[idx[i]] for i in range(num_classes)]
    label = [torch.ones(len(data[i])) * i for i in range(num_classes)]

    s = np.random.dirichlet(np.ones(num_classes) * alpha, num_users)
    data_dist = np.zeros((num_users, num_classes))

    for j in range(num_users):
        data_dist[j] = (
                (s[j] * images_per_client).astype('int') / (s[j] * images_per_client).astype('int').sum() *
                images_per_client).astype('int')
        data_num = data_dist[j].sum()
        data_dist[j][np.random.randint(low=0, high=num_classes)] += ((images_per_client - data_num))
        data_dist = data_dist.astype('int')

    X = []
    Y = []
    for j in range(num_users):
        x_data = []
        y_data = []
        for i in range(num_classes):
            if data_dist[j][i] != 0:
                d_index = np.random.randint(low=0, high=len(data[i]), size=data_dist[j][i])

                if dataset_name == "cifar10" or dataset_name == "cifar100":
                    x_data.append(torch.from_numpy(data[i][d_index]))
                else:
                    x_data.append(torch.unsqueeze(data[i][d_index],1))
                y_data.append(label[i][d_index])
        x_data = torch.cat(x_data).to(torch.float32)
        y_data = torch.cat(y_data).to(torch.int64)
        if dataset_name == "cifar10" or dataset_name == "cifar100":
            x_data = x_data.permute(0,3,1,2)
        X.append(x_data)
        Y.append(y_data)
    return X, Y


def get_datasets(dataset_name, data_iid=True, num_users=2, global_data=False, seed=42, images_per_client=500):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = None
    dataset_test = None
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                   std=[0.2023, 0.1994, 0.2010])])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                 std=[0.2023, 0.1994, 0.2010])])
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=False, transform=transform_test)
    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                   (0.2675, 0.2565, 0.2761))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                  (0.2675, 0.2565, 0.2761))])
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=False, transform=transform_test)
    elif dataset_name == "mnist":
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Pad(2),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.ToPILImage(),
                                             transforms.Pad(2),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('data/mnist', train=True, download=True, transform=transform_train)
        dataset_test = datasets.MNIST('data/mnist', train=False, download=False, transform=transform_test)
    elif dataset_name == "emnist":
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Pad(2),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.ToPILImage(),
                                             transforms.Pad(2),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.EMNIST('data/emnist', train=True, download=True,
                                        transform=transform_train, split='byclass')
        dataset_test = datasets.EMNIST('data/emnist', train=False, download=False,
                                       transform=transform_test, split='byclass')

    if global_data:
        if data_iid:
            iidtype = "iid"
        else:
            iidtype = "niid"
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        if not os.path.exists(f"dataset/{dataset_name}"):
            os.mkdir(f"dataset/{dataset_name}")
        if not os.path.exists(f"dataset/{dataset_name}/{iidtype}"):
            os.mkdir(f"dataset/{dataset_name}/{iidtype}")
        if not os.path.exists(f"dataset/{dataset_name}/{iidtype}/seed{seed}"):
            os.mkdir(f"dataset/{dataset_name}/{iidtype}/seed{seed}")
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/global_test.pkl",'wb') as f:
            cPickle.dump(dataset_test, f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/transform_train.pkl",'wb') as f:
            cPickle.dump(transform_train, f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/transform_test.pkl",'wb') as f:
            cPickle.dump(transform_test, f)
        return "done"

    if data_iid:
        imgs_train, labels_train = dirichlet(dataset=dataset_train, num_users=num_users,
                                             images_per_client=images_per_client, alpha=100, dataset_name=dataset_name)
        imgs_test, labels_test = dirichlet(dataset=dataset_test, num_users=num_users, images_per_client=images_per_client,
                                           alpha=100, dataset_name=dataset_name)
        iidtype = "iid"
    else:
        imgs_train, labels_train = dirichlet(dataset=dataset_train, num_users=num_users, images_per_client=images_per_client,
                                             alpha=0.1, dataset_name=dataset_name)
        imgs_test, labels_test = dirichlet(dataset=dataset_test, num_users=num_users, images_per_client=images_per_client,
                                           alpha=0.1, dataset_name=dataset_name)
        iidtype = "niid"

    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    if not os.path.exists(f"dataset/{dataset_name}"):
        os.mkdir(f"dataset/{dataset_name}")
    if not os.path.exists(f"dataset/{dataset_name}/{iidtype}"):
        os.mkdir(f"dataset/{dataset_name}/{iidtype}")
    if not os.path.exists(f"dataset/{dataset_name}/{iidtype}/seed{seed}"):
        os.mkdir(f"dataset/{dataset_name}/{iidtype}/seed{seed}")


    for i in tqdm(range(num_users)):
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/imgs_train_dev{i}.pkl", 'wb') as f:
            cPickle.dump(imgs_train[i], f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/labels_train_dev{i}.pkl", 'wb') as f:
            cPickle.dump(labels_train[i], f)

        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/imgs_test_dev{i}.pkl", 'wb') as f:
            cPickle.dump(imgs_test[i], f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/labels_test_dev{i}.pkl", 'wb') as f:
            cPickle.dump(labels_test[i], f)
    return "done"


def get_notification_transfer_done(connection, buffer_size, recv_timeout, target, verbose):
    """
    Function for getting notification that the transfer of files between the Server and the Client is done.
    Server receives message from the Client in the following format:

        {data}

        where

        {data} = {filename};{filesize}

    Returns
    -------
    filename : string
        The name of the file received.

    filesize : string
        The size of the file received.
    """
    msg = receive_msg(connection, buffer_size, recv_timeout, verbose)
    assert msg is not None, f"[!] Received no input from {target}"
    filename, filesize = msg.split(';')
    return filename, int(filesize)


def progress_bar(f, size, sent, p):
    progress = sent / size * 100.
    sys.stdout.write(f"({p[0]}:{p[1]}) {f}\'s progress: {progress}\r")


def scp_file(target_ip, target_port, target_usr, target_pwd, target_path, zip_filename, source_path, verbose):
    """
    File for sending a file through SCP.
    """
    if verbose:
        print(f"[+] Server is sending zip file {zip_filename} to the Client.")
    retry = True
    while retry:
        try:
            time.sleep(np.random.randint(2,6))
            policy = paramiko.client.AutoAddPolicy
            with paramiko.SSHClient() as client:
                client.set_missing_host_key_policy(policy)
                client.connect(target_ip, username=target_usr, password=target_pwd, port=22, auth_timeout=200, banner_timeout=200)

                with SCPClient(client.get_transport()) as scp:
                    scp.put(path.join(source_path, zip_filename), remote_path=target_path)
                retry = False

        except BaseException as e:
            print(f"[!] ERROR: {e}")
            print(f"[!] ERROR Connection failed. Could not connect to IP {target_ip} with username "
                  f"{target_usr} and password {target_pwd} for port {target_port}")
            print(f"[!] ERROR: could not put on {source_path} the file {zip_filename} for sending on the "
                  f"remote_path={target_path}")
            retry = True
            print(f"[!] Retrying...")
            time.sleep(5)
            # exit(-1)
    if verbose:
        print("[+] Server sent zip file to the Client.\n")


def unzip_file(connection, zip_filename, target_path, verbose):
    """
    Function for unzipping a file given as parameter. If the zip file cannot be extracted (errors occur), a "Resend"
    message is sent to the Client.
    """
    if verbose:
        print(f"[+] Unzipping file {zip_filename}")
    with zipfile.ZipFile(f"{path.join(target_path, zip_filename)}", 'r') as zip_ref:
        try:
            zip_ref.extractall(path=target_path)
        except BaseException as e:
            if verbose:
                print(f"[!] Encountered error when unzipping: {e}.")
            send_msg(connection, "Resend", verbose)
    if verbose:
        print(f"[+] Extracted file {zip_filename} to {target_path}\n")
        

def zip_file(filename, target_path, verbose):
    """
    Function for zipping a file.
    """
    zip_filename = filename.split(".")[0] + ".zip"
    if verbose:
        print(f"[+] Zipping the file {filename}")
    with zipfile.ZipFile(path.join(target_path, zip_filename), 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(path.join(target_path, filename), basename(path.join(target_path, filename)))
    if verbose:
        print(f"[+] File {zip_filename} zipped in {target_path}")
    filesize = os.path.getsize(path.join(target_path, zip_filename))
    if verbose:
        print(f"[+] File size of {zip_filename} is {filesize / 1000000} MB\n")
    return zip_filename, filesize


def close_connection(connection, verbose):
    """
    Function for closing the connection with the Server.
    """
    if verbose:
        print("[+] Closing the socket.")
    connection.close()
    if verbose:
        print("[+] Socket closed.\n")


def send_msg(connection, msg, verbose):
    """
    Function for sending a string message to the Client.
    """
    if verbose:
        print(f"[+] Server sending message \"{msg}\" to the Client.")
    msg = pickle.dumps(msg)
    connection.sendall(msg)
    if verbose:
        print(f"[+] Message sent.\n")


def receive_msg(connection, buffer_size, recv_timeout, verbose):
    """
    Function for receiving a string message and returning it.

    Returns
    -------
    subject : string
        Returns None if there was an error or if recv_timeout seconds passed with unresponsive Client.
        Returns the received message otherwise.
    """
    received_data, status = recv(connection, buffer_size, recv_timeout, verbose)
    if status == 0:
        connection.close()
        if verbose:
            print(f"[!] Connection closed either due to inactivity for {recv_timeout} seconds or due "
                  f"to an error.")
        return None

    if verbose:
        print(f"[+] Server received message from the Client: {received_data}\n")
    return received_data


def recv(connection, buffer_size, recv_timeout, verbose):
    """
    Function for receiving a string message and returning it.

    Returns
    -------
    received_data : string
        If there is no data received for recv_timeout seconds or if there is an exception returns None.
        If the message is received, it is decoded and returned

    status : int
        Returns 0 if the connection is no longer active and it should be closed.
        Returns 1 if the message was received successfully.
    """
    recv_start_time = time.time()
    received_data = b""
    while True:
        status = 0
        try:
            data = connection.recv(buffer_size)
            received_data += data

            if data == b"":  # Nothing received from the client.
                received_data = b""
                # If still nothing received for a number of seconds specified by the recv_timeout attribute, return
                # with status 0 to close the connection.
                if (time.time() - recv_start_time) > recv_timeout:
                    return None, status
            elif str(data)[-2] == '.':
                if verbose:
                    print(f"[+] All data ({len(received_data)} bytes) received.")

                if len(received_data) > 0:
                    try:
                        # Decoding the data (bytes).
                        received_data = pickle.loads(received_data)
                        # Returning the decoded data.
                        status = 1
                        return received_data, status

                    except BaseException as e:
                        if verbose:
                            print(f"[!] Error decoding the Client's data: {e}.\n")
                        return None, status
            else:
                # In case data is received from the client, update the recv_start_time to the current time to reset
                # the timeout counter.
                recv_start_time = time.time()

        except BaseException as e:
            if verbose:
                print(f"[!] Error receiving data from the Client: {e}.\n")
            return None, 0
