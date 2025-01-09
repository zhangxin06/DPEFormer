# -*- coding: utf-8 -*-
import os
import time
import argparse
import _thread
import random
import json
import socket
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
import os

import numpy as np
try:
    import torch
except ImportError:
    try:
        import tensorflow as tf
    except ImportError:
        print("No pytorch and tensorflow module")


def set_parser():
    parser = argparse.ArgumentParser(description='..')
    parser.add_argument('-p', '--proportion', type=float, default=0.8,
                        help='The ratio of gpu free memory to total memory')
    parser.add_argument('-n', '--gpu_nums', type=int, default=1,
                        help='The numbers of GPU to scramble')
    parser.add_argument('-t', '--times', type=int, default=1800,
                        help='Sleep time if scramble gpu')
    args = parser.parse_args()

    return args


def parse(qargs, results):
    result_np = []
    for line in results[1:]:
        result_np.append([''.join(filter(str.isdigit, word)) for word in line.split(',')])
    result_np = np.array(result_np)

    return result_np


def query_gpu():
    qargs = ['index', 'memory.free', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv, noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()

    return parse(qargs, results), results[0].strip()


class GPUManager(object):
    def __init__(self, args):
        self._args = args

    def choose_free_gpu(self):
        qresult, qindex = query_gpu()
        qresult = int(qresult)

        # if qresult.shape[0] == 0:
        #     print('No GPU, Check it.')
        # else:
        qresult_sort_index = np.argsort(-qresult[:, 1])
        idex = [i for i in qresult_sort_index if qresult[i][1]/qresult[i][2] > self._args.proportion]
        gpus_index = qresult[:, 0][idex]
        gpus_memory = qresult[:, 1][idex]
        return gpus_index, gpus_memory


def compute_storage_size(memory):
    return pow(memory * 1024 * 1024 / 8, 1/3) * 0.9


def worker(gpus_id, size):
    try:
        a = torch.zeros([size, size, size], dtype=torch.double, device=gpus_id)
        while True:
            torch.mul(a[0], a[0])
            if random.random() > 0.5:
                time.sleep(0.0000001)
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_id)
        a = tf.zeros([size, size, size], dtype=tf.dtypes.float64)
        while True:
            tf.matmul(a[0], a[0])
            if random.random() > 0.5:
                time.sleep(0.0000001)


class EmailSender(object):
    def __init__(self, host_server, user, pwd, sender):
        self.host_server = host_server
        self.user = user
        self.pwd = pwd
        self.sender = sender

    def send_email(self, receiver, subject, content):
        receiver = [receiver] if isinstance(receiver, str) else receiver
        message = MIMEText(content, 'plain', 'utf-8')
        message['Subject'] = subject
        message['From'] = self.sender
        message['To'] = ", ".join(receiver)

        try:
            smtp_obj = SMTP_SSL(self.host_server)
            smtp_obj.ehlo(self.host_server)
            smtp_obj.login(self.user, self.pwd)
            smtp_obj.sendmail(self.sender, receiver, message.as_string())
            smtp_obj.quit()
            print("The mail was sent successfully.")
        except Exception as e:
            print(e)


def main(args, ids):
    # with open(args.email_conf, "r") as f:
    #     email_conf = json.load(f)
    # email_sender = EmailSender(email_conf['host'],
    #                            email_conf['user'],
    #                            email_conf['pwd'],
    #                            email_conf['sender'])

    gpu_manager = GPUManager(args)
    gpus_free, gpus_memory = gpu_manager.choose_free_gpu()

    if len(gpus_free) == 0:
        # print('No free GPUs, waiting for someone else to release.')
        pass
    else:
        sca_nums = args.gpu_nums - len(ids)
        if sca_nums > 0:
            sizes = [int(compute_storage_size(i)) for i in gpus_memory]
            for gpus_id, size in zip(gpus_free[:sca_nums], sizes[:sca_nums]):
                ids.append(gpus_id)
                print("Scramble GPU {}".format(gpus_id))
                _thread.start_new_thread(worker, (gpus_id, size))
                time.sleep(30)

            hostname = socket.gethostname()
            gpu_ids = ', '.join(gpus_free[:sca_nums].astype('str'))
            # subject = f"{hostname}: GPU {gpu_ids} has been scrambled"
            # content = f"{hostname}: GPU {gpu_ids} has been scrambled, and will be released in {args.times//60} minutes!"
            # email_sender.send_email(email_conf['receiver'], subject, content)


if __name__ == '__main__':
    ids = []
    args = set_parser()
    while True:
        main(args, ids)
        if len(ids) >= args.gpu_nums:
            time.sleep(args.times)
            break
