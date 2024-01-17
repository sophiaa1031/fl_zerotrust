#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


def saveData(my_list, file_path):
    try:
        # 将列表元素转换为字符串并以换行符分隔
        list_as_str = '\n'.join(map(str, my_list))

        # 打开文件并写入列表内容
        with open(file_path, 'w') as file:
            file.write(list_as_str)

        print(f"列表已成功保存到文件: {file_path}")
    except Exception as e:
        print(f"保存列表到文件时发生错误: {str(e)}")


def mainFunction(args):
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users, args)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    # dict_data_ratio = {key: len(value) / sum(len(val) for val in dict_users.values()) for key, value in dict_users.items()}
    dict_data_ratio_list = [len(value) / sum(len(value) for value in dict_users.values()) for value in
                            dict_users.values()]

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    idxs_users = np.arange(args.num_users)  # 保证第一个用户是恶意用户
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        if args.attack and args.defense != 'none' and iter == args.attack_epoch:  # 去掉恶意用户, 如果大于args.epochs，则都不去掉
            if args.defense == 'zerotrust':
                print('change the ratio...')
                dict_users[0] = np.random.choice(dict_users[0], 600, replace=True)
                dict_data_ratio_list = [1 / (args.num_users)] * (args.num_users)
                print('dict_data_ratio_list',dict_data_ratio_list)
            if args.defense == 'remove':  # 去掉恶意用户, 如果大于args.epochs，则都不去掉
                print('remove a user...')
                idxs_users = np.arange(args.num_users - 1)
                # dict_data_ratio = {key: value for key, value in dict_data_ratio.items() if key != list(dict_data_ratio.keys())[0]}
                dict_data_ratio_list = [1 / (args.num_users - 1)] * (args.num_users - 1)
                print('dict_data_ratio_list',dict_data_ratio_list)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(idx, iter, net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals, dict_data_ratio_list)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = np.sum([loss_locals[i] * dict_data_ratio_list[i] for i in range(len(loss_locals))])
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # save data
    folder_path = 'plot_data/mnist_noniid_loss/'
    if not args.attack:
        file_path = folder_path + "benign.txt"
    elif args.defense == 'none':
        file_path = folder_path + "attack" + "_" + str(args.attack_epoch) + ".txt"
    else:
        file_path = folder_path + args.defense + "_" + str(
            args.attack_epoch) + ".txt"  # 指定文件路径(remove_malicious, attack, all_benign,attack_middle)
    print(file_path)
    # saveData(loss_train, file_path)

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig(
    #     './save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.noniid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print('device:{}'.format(args.device))

    # 为了debug
    # args.attack = True
    # args.attack_epoch =1
    # args.defense = 'remove'
    mainFunction(args)
