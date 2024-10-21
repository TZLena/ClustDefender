import numpy as np
import torch
import torch.optim as optim

from matplotlib import pyplot as plt
from sklearn import svm

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, accuracy_score


class UserAggregator:
    '''Aggregate per-user gradiant'''

    def __init__(self, args, model_cls, device):
        self.args = args
        self.model = model_cls(args).to(device)
        #print("args", model_cls(args))
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self._init_grad_param_vecs()

        self.direction = torch.zeros(self.count_p()).to(device)
        #self.weight = torch.ones(50).to(self.device)  # len(users)
        self.susp = torch.zeros(50).to(self.device)

        self.data_list = []
        self.label_list = []

        self.near_dist = 0.7 * self.args.eps
        self.eps = self.args.eps
        self.min_samples = self.args.min_samples

        #self.benign = 0
        #self.poison = 0

        self.svc = svm.SVC(kernel='linear')



    def count_p(self):  # flair
        P = 0
        for param in self.model.parameters():
            if param.requires_grad:
                P = P + param.nelement()
        print("p",P)
        return P

    def _init_grad_param_vecs(self):
        self.user_grad = {}
        self.grad_list = []
        self.label = []
        self.user_sample_num = None

        self.optimizer.zero_grad()


    def update(self, all_sample_num,step,root_user_index):

        self.update_model_grad(all_sample_num,step,root_user_index)
        self._init_grad_param_vecs()  # 每轮更新后重置

    def update_model_grad(self, all_sample_num, step, root_user_index):  #多加了两个变量
        #print("进入聚合")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = torch.sum(
                    self.user_grad[name] / all_sample_num,  # 原始    self.user_grad[name] / sum(self.user_sample_num.tolist())
                    dim=0,
                ).cuda()
        #print("all_sample_num:",all_sample_num)
        #print("user_sample_num sum:",sum(self.user_sample_num.tolist()))
        #print("agg grad norm",[torch.norm(param).item() for param in self.model.parameters() if param.requires_grad])
        #print("agg grad norm sum",torch.norm(torch.cat([param.view(-1) for param in self.model.parameters() if param.requires_grad])))
        #需要裁剪梯度or
        self.optimizer.step()
        #print("离开聚合")

    def collect(self, user_grad, user_sample_num):
        for name in user_grad:
            if name not in self.user_grad:
                self.user_grad[name] = user_grad[name]
            else:
                self.user_grad[name] += user_grad[name]

        if self.user_sample_num is None:
            self.user_sample_num = user_sample_num
        else:
            self.user_sample_num += user_sample_num

    def collect_by_uindex(self, user_grad, user_sample_num, uindex):
        print("uindex", uindex)
        assert len(self.user_grad) != 0, "collect_by_uindex cannot apply to empty user_grad!"
        for name in user_grad:
            self.user_grad[name][uindex] += user_grad[name]
        self.user_sample_num[uindex] += user_sample_num


    def collect_grad_list(self,users,uindex):
        #print("len(users)", len(users))
        for i in range(len(users)):
            users_param = []
            for name in self.user_grad:
                user_param = self.user_grad[name][i]
                users_param.append(user_param)
            self.grad_list.append(users_param)
            if i in uindex:
                self.label.append(1)
            else: self.label.append(0)

    def collect_grad_list_label(self, users, uindex):
        #print("collect_grad_list_label")
        name = 'user_encoder.multihead_attention.W_Q.weight'
        ben_count = 0

        for i in range(len(users)):
            if i in uindex:
                #print("mal grad;",self.user_grad[name][i])
                self.label_list.append(1)
                self.data_list.append(self.user_grad[name][i])
            else:
                if ben_count < 50:
                    self.label_list.append(0)
                    self.data_list.append(self.user_grad[name][i])
                    ben_count += 1
                else:
                    continue  # 跳过多余的标签为0的用户



    '''def collect_grad_list_for_view(self,users,uindex):
        print("uindex", uindex)
        grad_list = []
        mal_list = []
        print("len(users)", len(users))
        for i in range(len(users)):
            users_param = []
            for name in self.user_grad:
                user_param = self.user_grad[name][i]
                users_param.append(user_param)
            grad_list.append(users_param)
        self.grad_list = grad_list
        for i in uindex:
            mal_list.append(self.grad_list[i])

        return mal_list'''

'''
ua中的user_grad:由模型名和所有用户参数值的字典组成，{name1:[user[0],user[1]...],name2:...,...,name10:...}
fl中的grad_list:每个用户的所有参数按顺序堆叠组成[[user[0][name0],...,user[0][name10]],...,[user[-1][name0],...,user[-1][name10]]]'''
