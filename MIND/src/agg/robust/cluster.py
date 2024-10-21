import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import DBSCAN

from MIND.src.agg.user import UserAggregator


class ClustAggregator(UserAggregator):


    def grad_cluster(self,step):
        data = np.array([grad[5].flatten().numpy() for grad in self.grad_list])  # news_encoder_WQ_weight
        dbscan = DBSCAN(eps = self.args.eps, min_samples= self.args.min_samples).fit(data)  # 动态调整eps值
        labels = dbscan.labels_
        #print("聚类结果：",labels)
        positions = [index for index, value in enumerate(labels) if value == -1]
        print("噪声点位置",positions)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("聚类数量：",n_clusters)

        if np.count_nonzero(labels == -1) > 10:
            self.args.eps += 0.1
        if np.count_nonzero(labels == -1) < 5:
            self.args.eps -= 0.1
        if n_clusters == 0:
            self.args.eps = 3.0
            self.args.min_samples = 5
        if n_clusters > 1:
            self.args.min_samples += 1


        # 将 labels 转换为二元标签（噪声点为 1，非噪声点为 0）
        binary_labels = [1 if label == -1 else 0 for label in labels]
        # 计算混淆矩阵
        if len(np.unique(labels)) == 2:
            tn, fp, fn, tp = confusion_matrix(self.label, binary_labels).ravel()
        else: tn, fp, fn, tp = 50,0,0,0
        print("tn, fp, fn, tp",tn, fp, fn, tp)
        # 计算正确分离的恶意客户端和被错误分离的良性客户端比例
        if (tp + fn) == 0:
            recall = 0
        else: recall = tp / (tp + fn)  # 正确分出来的投毒客户端占所有投毒比例

        specificity = fp / (fp + tn)  #实际为良性点但没有被聚类的客户端占所有良性的比例

        print(f"第{step}个step的recall:",recall)
        print("specificity:",specificity)

        del_mask = torch.tensor(labels) != -1  # 删除labels=-1的噪声点，留下可聚合的点
        #print("del_mask",del_mask)

        # 删除掩码为1的客户端
        for name in self.user_grad:
            self.user_grad[name] = self.user_grad[name][del_mask]
        self.user_sample_num = self.user_sample_num[del_mask]

    def grad_cluster_v2(self, step, root_user_index):  # grad_list和user_grad间转换

        data = np.array([grad[5].flatten().numpy() for grad in self.grad_list])  # news_encoder_WQ_weight
        distance = [np.linalg.norm(d - data[root_user_index]) for d in data]  # 每个点与根的距离

        near_list = [grad for i, grad in enumerate(self.grad_list) if  # 筛选与标准店距离在阈值内的用户，权重按1计算
                              distance[i] < self.near_dist]
        print("邻近点数量:",len(near_list))
        near_avg = []

        for name in range(len(near_list[0])):  # 计算邻近点的均值，作为标准计算后续的权重
            param = torch.stack([grad[name] for grad in near_list],dim=0)  # 每个用户同一个参数的值
            avg_param = torch.mean(param,dim=0)
            near_avg.append(avg_param)

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data)  # 动态调整eps值
        labels = dbscan.labels_
        print("聚类结果：",labels)
        positions = [index for index, value in enumerate(labels) if value == -1]
        print("噪声点位置", positions)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("聚类数量：", n_clusters)

        max_eps, min_eps = 5.0, 2.5
        if np.count_nonzero(labels == -1) > 10:
            self.eps = min(self.eps + 0.1, max_eps)
        if np.count_nonzero(labels == -1) < 5:
            self.eps = max(self.eps - 0.1, min_eps)
        if n_clusters == 0:
            self.eps = 3.0
            self.min_samples = 5
        if n_clusters > 1:
            self.min_samples += 1
        print("聚合后eps",self.eps,"min_samples",self.min_samples)

        if labels[root_user_index] != -1:  # 根被聚类
            mid_list = [self.grad_list[i] for i, label in enumerate(labels) if label == labels[root_user_index] and not any(
                torch.equal(self.grad_list[i][5], near_grad[5]) for near_grad in near_list)]  # 距离中等的点

        else:  # 根是噪声点
            mid_list = [self.grad_list[i] for i, label in enumerate(labels) if label != -1 and not any(
                torch.equal(self.grad_list[i][5], near_grad[5]) for near_grad in near_list)]

        print("中间点数量:",len(mid_list))
        mid_dist = []
        for client_grad in mid_list:  # 计算每个用户与邻近均值点间的距离 [torch.tensor(0.0),...,torch.tensor(0.0)]
            norms = [torch.norm(client_grad[i] - near_avg[i]) for i in range(len(client_grad))]  # 每个参数的范数
            norm = torch.sum(torch.stack(norms),dim=0)
            mid_dist.append(torch.tensor(norm))
        if len(mid_list) > 0:
            mid_dist_sum = torch.sum(torch.stack(mid_dist), dim=0)
            epsilon = 1e-8
            mid_weight = [dist / (mid_dist_sum + epsilon) for dist in mid_dist]  # 求出每个mid中用户的权重
            print("mid_weight:",mid_weight)
        else:
            mid_weight = []

        grad_agg = []
        for name in range(len(mid_list[0])):  # name
            if len(mid_weight) == 0:
                sum = near_avg[name]
                grad_agg.append(sum)
            else:
                mid_agg = torch.sum(torch.stack([client_grad[name] * mid_weight[index] for index,client_grad in enumerate(mid_list)],dim=0),dim=0)  # mid加权聚合
                sum = (len(mid_list) / (len(mid_list) + len(near_list))) * mid_agg + (len(near_list) / (len(mid_list)+len(near_list))) * near_avg[name]
                grad_agg.append(sum)

        return grad_agg

    def update_model_grad(self, all_sample_num, step, root_user_index):
        print("cluster agg")

        grad_agg = self.grad_cluster_v2(step,root_user_index)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                param.grad = grad_agg[i].cuda()
        self.optimizer.step()

