import os
from tqdm import tqdm
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models import CNNMnist
from utils import get_dataset, ALU_aggregation_weight, get_local_time, get_transmission_rate
from cluster import cluster
from BA import Bandwidth_Allocation
from devices import Edge_Device
import argparse
import random
import csv
from schedule_policy import schedule

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=500,
                        help="number of rounds of training")
    parser.add_argument('--num_devices', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=6,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    
    # model arguments
    parser.add_argument('--model', type=str, default='CNN', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 2 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--stale', type=int, default=5, help='staleness ')
    args = parser.parse_args()
    return args
def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=1024,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss/len(testloader)
a = 0.8 #年龄模型超参数
if __name__=='__main__':
    args = args_parser()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    #加载数据集
    train_dataset, test_dataset, user_groups = get_dataset(args)

    #创建全局模型
    if args.dataset == 'mnist':
        global_model = CNNMnist(args)
        ## 加载初始模型
        global_model.load_state_dict(torch.load("./save/origin_model.pth"))

    global_model.to(device)

    #复制全局模型参数
    global_weights = global_model.state_dict()
    #边缘设备
    devices = []
    #lables向量
    vectors = []

    #边缘设备传输能力
    channel_cap = get_transmission_rate(args.num_devices)
    #完成本地训练的总时间
    whole_times = get_local_time(args.num_devices)  
    #每个边缘设备训练完成还需要多少时间
    remain_times = copy.deepcopy(whole_times)              
    for id in range(args.num_devices):
        ds = Edge_Device(args, id, user_groups[id], train_dataset)
        devices.append(ds)
        vectors.append(ds.get_vector())
    
    #聚类
    clusters, clusters_num = cluster(vectors)
    #边缘设备分类
    for id in range(args.num_devices):
        devices[id].kind = clusters[id]
    print("Devices are ready")
    #此轮被选择的边缘设备
    choice_devices = []  
    #初始化簇类
    devices_kind = []    
    for i in range(clusters_num):
        devices_kind.append([])
    #本地更新年龄
    devices_age = []
    #每轮训练时间
    epochs_time = [0]   
     #每轮正确率
    epochs_acc =  [0]  
    #每轮平均年龄
    epochs_age = [0]
    #已经做好上传准备的设备
    current_devices = []  
    #准备上传的信息
    ready_weights = {}
    ready_losses = {}
    #最大聚合设备数量
    R = max(int(args.frac * args.num_devices), 1) 
    #上一轮是否参加更新
    last_update = []      


    num_devices = args.num_devices
    for idx in range(num_devices):
        #下载全局模型
        devices[idx].weight_download(0, copy.deepcopy(global_weights) )
        #开始本地训练
        devices[idx].local_train()
        last_update.append(0)
    #第一轮需要等待
    sorted_time = sorted(whole_times)
    T = sorted_time[20]
    update_age = {} #上传模型的年龄
    for idx in range(num_devices):
        remain_times[idx] = remain_times[idx] - T
        #完成本地训练，存储上传信息
        if remain_times[idx] <= 0:
            device = devices[idx]
            devices_kind[device.kind].append(idx)
            loss = device.send_loss()
            weights = device.send_weight()
            current_devices.append(idx)
            ready_weights[idx] = copy.deepcopy(weights)
            ready_losses[idx] = loss
            update_age[idx] = device.global_age
    #每轮时间
    epochs_time[0] = T
    #年龄敏感参数
    ALU = {} 
    #年龄容忍度
    stale = args.stale
    select_times = []
    for i in range(num_devices):
        select_times.append(1)
    #全局训练
    for round in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {round} |\n')
        global_model.train()
        aggregation_idx = []

        devices_kind1 = copy.deepcopy(devices_kind)
        #删除超过容忍度的信息
        for i in range(10):
            for idx in devices_kind[i]:
                if round - update_age[idx] > stale:
                    devices_kind1[i].remove(idx)
        devices_kind = copy.deepcopy(devices_kind1)
        print(devices_kind)

        r = R

        #调度设备
        aggregation_idx = schedule(devices_kind, last_update, ready_losses, select_times, R)

        print(aggregation_idx)

        # 带宽分配
        T = Bandwidth_Allocation(aggregation_idx, channel_cap)
        print("传输时间 = ", T)

        ## 年龄权重模型
        total_ALU = 0
        total_age = 0
        for idx in aggregation_idx:
            total_age += round - update_age[idx]
            ALU[idx] = a**(round - update_age[idx])
            total_ALU = total_ALU + ALU[idx]
        avg_age = total_age / R
        epochs_age.append(avg_age)
        print("Avg_age =", avg_age)

        ww = {}
        for idx in aggregation_idx:
            ww[idx] = ALU[idx] / total_ALU

        # 聚合权重
        g_weights = ALU_aggregation_weight(ww, aggregation_idx, ready_weights)
        for key in global_weights.keys():
            global_weights[key] = global_weights[key] + g_weights[key]

        ## 更新全局模型
        global_model.load_state_dict(global_weights)    
        #此轮延时
        epochs_time.append(epochs_time[-1] + T)     
        #测试集acc,loss
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        epochs_acc.append(test_acc)
        print("Round = ", round, "Accuracy = ", test_acc, "Loss = ", test_loss, "Time = ", epochs_time[-1])

        #标记未被选用过
        for idx in range(num_devices):
            last_update[idx] = 0
        
        # 标记上一轮是否已经被聚合过,次数加1
        for idx in aggregation_idx:
            last_update[idx] = 1
            select_times[idx] += 1

        # 从准备集中删除被聚合过的客户
        for idx in aggregation_idx:
            devices[idx].send_nonempty = 0
            devices_kind[devices[idx].kind].remove(idx)

        ## 通信时，其他设备继续训练
        for idx in range(num_devices):
            if idx in aggregation_idx:
                continue
            remain_times[idx] = remain_times[idx] - T
            if remain_times[idx] <= 0 :#在周期内能完成本地训练
                device = devices[idx]
                loss = device.send_loss()
                weights = device.send_weight()
                ready_weights[idx] = copy.deepcopy(weights)
                ready_losses[idx] = loss
                if idx not in devices_kind[device.kind]:
                    devices_kind[device.kind].append(idx)
                current_devices.append(idx)
                update_age[idx] = device.global_age

        for idx in current_devices:
            if devices[idx].nonempty == 1:
                devices[idx].local_train()
                remain_times[idx] = max(remain_times[idx] + whole_times[idx], 0)
            else :
                remain_times[idx] = whole_times[idx]
        total_ALU = 0
        
        # 下载全局模型
        for idx in range(num_devices):
            devices[idx].weight_download(round + 1, global_weights)
        # 开始新的本地训练
        for idx in current_devices:
            devices[idx].local_train()
        current_devices = []
        with open('./save/Semi-AFL_iid{}.csv'.format(args.iid), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round, test_acc, test_loss, epochs_time[-1], avg_age])

    n = 0
    

    end = time.time()


