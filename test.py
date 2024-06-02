import os
gpus = [0,1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import time
import datetime
import scipy.io
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
from sklearn import metrics
cudnn.benchmark = False
cudnn.deterministic = True

class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.root = './.../'

        self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = MCTD().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.model, (1, 22, 1000))


    def get_source_data(self):

        # train data
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def test(self):

        test_data, test_label = self.get_source_data()


        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        bestkappa = 0
        averkappa = 0
        bestrecall = 0
        averrecall = 0
        bestf1 = 0
        averf1 = 0
        bestpreci = 0
        averpreci = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        path_ours = f'./ours/oursweight/2a_i{self.nSub + 1}.pth'


        # test process
        toutput = 0
        alltok = 0

        self.model.eval()
        self.model.load_state_dict(torch.load(path_ours, map_location=torch.device('cpu')))
        Toks, toutputs = self.model(test_data)
        for tout in toutputs:
            tout = tout.float()
            toutput += tout.detach()
        for Tok in Toks:
            Tok = Tok.float()
            alltok += Tok.detach()
        Cls = toutput


        loss_test = self.criterion_cls(Cls, test_label)
        y_pred = torch.max(Cls, 1)[1]
        kappa = metrics.cohen_kappa_score(test_label.cpu().data.numpy(), y_pred.cpu().data.numpy())
        recall = metrics.recall_score(test_label.cpu().data.numpy(), y_pred.cpu().data.numpy(), average='macro')
        f1 = metrics.f1_score(test_label.cpu().data.numpy(), y_pred.cpu().data.numpy(), average='macro')
        preci = metrics.precision_score(test_label.cpu().data.numpy(), y_pred.cpu().data.numpy(), average='macro')
        acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))


        return acc, kappa, recall, f1, preci, Y_true, Y_pred
        # writer.close()


def main():
    best = 0
    aver = 0
    result_write = open("./results/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()

        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i + 1))
        exp = ExP(i + 1)

        acc, kappa, recall, f1, preci, Y_true, Y_pred = exp.train()
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(acc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best recall is: ' + str(recall) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best f1 is: ' + str(f1) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best preci is: ' + str(preci) + "\n")
        result_write.close()

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))
