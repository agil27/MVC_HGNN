from dataset import DEAPLoader
from config import config
from utils import *
from random import shuffle
import torch
from torch import nn, optim, Tensor
from copy import deepcopy
from models import HGNN, MVHGNN, MVCHGNN, MVTSHGNN
import sys

class DEAPTrainer:
    def __init__(self, gpu=0):
        super(DEAPTrainer, self).__init__()
        self.cfg = config()
        self.loader = DEAPLoader()
        self.device = self.cfg.device
        self.gpu = gpu
        self.n_sample = self.loader.num_of_samples
        self.n_test = int(self.n_sample * self.cfg.test_rate)
        self.n_fold = self.cfg.n_fold
        self.n_class = self.cfg.class_num
        self.lr = self.cfg.lr
        self.n_epoch = self.cfg.n_epoch
        self.weight_decay = self.cfg.weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.log = self.cfg.log
        self.n_hid = self.cfg.n_hid
        self.drop_out = self.cfg.drop_out
        self.print_to_screen = self.cfg.print_to_screen
        self.milestone = self.cfg.milestone
        self.gamma = self.cfg.gamma
        self.k_list = [2, 3, 4, 5, 6]

        if self.device == 'cuda':
            torch.cuda.set_device(self.gpu)

    def _train_model(self, model, xs, label, idx_train, idx_test, ForwardParams):
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestone, gamma=self.gamma)
        best_acc = 0
        with open(self.log, 'w') as f:
            for epoch in range(self.n_epoch):
                if (epoch % 100 == 99):
                    f.write(f'---------------epoch {epoch + 1}---------------\n')
                    if self.print_to_screen:
                        print(f'---------------epoch {epoch + 1}---------------')
                scheduler.step()

                # @训练阶段
                model.train()
                running_loss, running_corrects = 0.0, 0
                feature, target = xs, label
                index = idx_train
                # 前向传播
                optimizer.zero_grad()
                score = model.forward(feature, ForwardParams)
                _, predicted = torch.max(score.data, 1)
                # 反向传播
                loss = self.criterion(score[index], target[index])
                loss.backward()
                optimizer.step()
                # 训练数据
                running_loss = running_loss + loss.item() * len(index)
                running_corrects = running_corrects + int(torch.sum(predicted[index] == label[index]))
                train_acc = float(running_corrects) / float(index.shape[0])
                if epoch % 100 == 99:
                    if self.print_to_screen:
                        print('train loss: %.3f' % running_loss)
                        print('train acc: %.3f' % train_acc)
                    f.write('train loss: %.3f\n' % running_loss)
                    f.write('train acc: %.3f\n' % train_acc)

                # @测试阶段
                model.eval()
                running_loss, running_corrects = 0.0, 0
                index = idx_test
                score = model.forward(feature, ForwardParams)
                _, predicted = torch.max(score.data, 1)
                loss = self.criterion(score[index], target[index])
                running_loss = running_loss + loss.item() * len(index)
                running_corrects = running_corrects + int(torch.sum(predicted[index] == label[index]))
                test_acc = float(running_corrects) / float(index.shape[0])
                if epoch % 100 == 99:
                    if self.print_to_screen:
                        print('test loss: %.3f' % running_loss)
                        print('test acc: %.3f' % test_acc)
                    f.write('test loss: %.3f\n' % running_loss)
                    f.write('test acc: %.3f\n' % test_acc)

                # 更新最优训练accuracy
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_wts = deepcopy(model.state_dict())

            model.load_state_dict(best_wts)
            f.write(f'best accuracy: {best_acc}')
        return model, best_acc

    def train_fc(self, task):
        # 生成所需数据和模型
        fts, H_list = self.loader.load_DEAP(ft_in_use=[0, 2, 3], k_list=self.k_list)
        labels = self.loader.label
        n_modal = len(fts)
        label = Tensor(labels[task]).long().to(self.device)
        H_list = [H[task] for H in H_list]
        G_list = generate_G_from_H(H_list)
        G_list = [Tensor(G).to(self.device) for G in G_list]
        xs = [Tensor(fts[i][task]).to(self.device) for i in range(n_modal)]
        in_ft_list = [x.shape[1] for x in xs]
        n_class = self.n_class
        model = MVHGNN(
            n_class=n_class,
            n_hid=self.n_hid,
            dropout=self.drop_out,
            in_ft_list=in_ft_list,
        )
        if self.device == 'cuda':
            model.cuda_()
        model = model.to(self.device)
        # 随机确定测试集和训练集
        indices = np.array(range(self.n_sample))
        for i in range(self.n_fold):
            shuffle(indices)
            idx_test, idx_train = indices[:self.n_test], indices[self.n_test:]
            self.log = self.cfg.log + 'fc_' + task + '_' + str(i) + '.txt'
            model, best_acc = self._train_model(model, xs, label, idx_train, idx_test, G_list)

    def train_kernel(self, task):
        x, H = self.loader.load_line_kernel()
        label = self.loader.label
        label = Tensor(label[task]).long().to(self.device)
        H = H[task]
        G = generate_G_from_H(H)
        G = Tensor(G).to(self.device)
        x = Tensor(x[task]).to(self.device)
        in_ft = x.shape[1]
        n_class = self.n_class
        model = HGNN(
            in_ch=in_ft,
            n_class=n_class,
            n_hid=self.n_hid,
            dropout=self.drop_out,
        )
        model = model.to(self.device)
        # 随机确定测试集和训练集
        indices = np.array(range(self.n_sample))
        for i in range(self.n_fold):
            shuffle(indices)
            idx_test, idx_train = indices[:self.n_test], indices[self.n_test:]
            self.log = self.cfg.log + 'kernel_' + task + '_' + str(i) + '.txt'
            model, best_acc = self._train_model(model, x, label, idx_train, idx_test, G)

    def train_ensemble(self, task):
        # 生成所需数据和模型
        fts, H_mat = [], []
        for k in self.k_list:
            fts_k, H_k = self.loader.load_DEAP(ft_in_use=[0, 2, 3], k_list=self.k_list)
            fts.append(fts_k)
            H_mat.append(H_k)
        labels = self.loader.label
        label = Tensor(labels[task]).long().to(self.device)
        H_mat = [[H[task] for H in H_vec] for H_vec in H_mat]
        G_mat = generate_G_from_H(H_mat)
        G_mat = [[Tensor(G).to(self.device) for G in G_vec] for G_vec in G_mat]
        xs = [[Tensor(x[task]).to(self.device) for x in x_vec] for x_vec in fts]
        in_ft_mat = [[x.shape[1] for x in x_vec] for x_vec in xs]
        n_class = self.n_class
        model = MVCHGNN(
            n_class=n_class,
            n_hid=self.n_hid,
            dropout=self.drop_out,
            in_ft_mat=in_ft_mat
        )
        if self.device == 'cuda':
            model.cuda_()
        model = model.to(self.device)
        # 随机确定测试集和训练集
        indices = np.array(range(self.n_sample))
        for i in range(self.n_fold):
            shuffle(indices)
            idx_test, idx_train = indices[:self.n_test], indices[self.n_test:]
            self.log = self.cfg.log + 'ensemble_' + task + '_' + str(i) + '.txt'
            model, best_acc = self._train_model(model, xs, label, idx_train, idx_test, G_mat)

    def train_cascade(self, task):
        # 生成所需数据和模型
        fts, H_mat = [], []
        for m in [0, 2, 3]:
            fts_m, H_m = self.loader.load_DEAP(ft_in_use=[m], k_list=self.k_list)
            fts.append(fts_m)
            H_mat.append(H_m)
        labels = self.loader.label
        label = Tensor(labels[task]).long().to(self.device)
        H_mat = [[H[task] for H in H_vec] for H_vec in H_mat]
        G_mat = generate_G_from_H(H_mat)
        G_mat = [[Tensor(G).to(self.device) for G in G_vec] for G_vec in G_mat]
        xs = [[Tensor(x[task]).to(self.device) for x in x_vec] for x_vec in fts]
        in_ft_mat = [[x.shape[1] for x in x_vec] for x_vec in xs]
        n_class = self.n_class
        model = MVCHGNN(
            n_class=n_class,
            n_hid=self.n_hid,
            dropout=self.drop_out,
            in_ft_mat=in_ft_mat
        )
        if self.device == 'cuda':
            model.cuda_()
        model = model.to(self.device)
        # 随机确定测试集和训练集
        indices = np.array(range(self.n_sample))
        for i in range(self.n_fold):
            shuffle(indices)
            idx_test, idx_train = indices[:self.n_test], indices[self.n_test:]
            self.log = self.cfg.log + 'cascade_' + task + '_' + str(i) + '.txt'
            model, best_acc = self._train_model(model, xs, label, idx_train, idx_test, G_mat)

    def train_twostep(self, task):
        # 生成所需数据和模型
        fts, H_list = self.loader.load_DEAP(ft_in_use=[0, 2, 3], k_list=self.k_list)
        labels = self.loader.label
        n_modal = len(fts)
        label = Tensor(labels[task]).long().to(self.device)
        H_list = [H[task] for H in H_list]
        Param_list = generate_params_from_H(H_list)
        Param_list = [(Tensor(EP).to(self.device), Tensor(NP).to(self.device)) for (EP, NP) in Param_list]
        xs = [Tensor(fts[i][task]).to(self.device) for i in range(n_modal)]
        in_ft_list = [x.shape[1] for x in xs]
        n_class = self.n_class
        model = MVTSHGNN(
            n_class=n_class,
            n_hid=self.n_hid,
            dropout=self.drop_out,
            in_ft_list=in_ft_list,
        )
        if self.device == 'cuda':
            model.cuda_()
        model = model.to(self.device)
        # 随机确定测试集和训练集
        indices = np.array(range(self.n_sample))
        for i in range(self.n_fold):
            shuffle(indices)
            idx_test, idx_train = indices[:self.n_test], indices[self.n_test:]
            self.log = self.cfg.log + 'twostep_' + task + '_' + str(i) + '.txt'
            model, best_acc = self._train_model(model, xs, label, idx_train, idx_test, Param_list)


def main():
    arg1, arg2, arg3 = sys.argv[1], sys.argv[2], sys.argv[3]
    gpu = int(arg3)
    mytrainer = DEAPTrainer(gpu=gpu)
    if arg2 == '-v':
        task = 'Valence'
    if arg2 == '-a':
        task = 'Arousal'

    if arg1 == '-e':
        mytrainer.train_ensemble(task)
    if arg1 == '-f':
        mytrainer.train_fc(task)
    if arg1 == '-c':
        mytrainer.train_cascade(task)
    if arg1 == '-k':
        mytrainer.train_kernel(task)
    if arg1 == '-t':
        mytrainer.train_twostep(task)

if __name__ == '__main__':
    main()


