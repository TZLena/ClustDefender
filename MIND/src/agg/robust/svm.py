import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

from MIND.src.agg.user import UserAggregator


class SVMAggregator(UserAggregator):

    def grad_classify(self):

        data = np.array([grad.flatten().numpy() for grad in self.data_list])
        label_list = np.array(self.label_list)
        print("shape: ",data.shape,label_list.shape)
        pca = PCA(n_components = 2)
        lda = LDA(n_components = 1)
        tsne = TSNE(n_components = 2)
        grad_dim = pca.fit_transform(data)
        #grad_dim = np.c_[grad_dim, np.zeros(grad_dim.shape[0])]

        self.svc.fit(grad_dim, label_list)
        sv = self.svc.support_vectors_  # 获取支持向量

        # 绘制数据点和支持向量
        plt.scatter(grad_dim[:, 0], grad_dim[:, 1], c=label_list, cmap=plt.cm.Paired)
        plt.scatter(sv[:, 0], sv[:, 1], facecolors='none', edgecolors='k', s=100)

        # 绘制决策边界
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.svc.decision_function(xy).reshape(XX.shape)

        # 绘制决策边界和平行于决策边界的线（支持向量）
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        plt.title('SVM Decision Boundary with PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

        w = self.svc.coef_[0]
        b = self.svc.intercept_[0]

        print(f"超平面: {w[0]:.2f} * x1 + {w[1]:.2f} * x2 + {b:.2f} = 0")

    def grad_pred(self):
        data = [grad[11].flatten().numpy() for grad in self.grad_list]  # user_encoder_WQ_weight
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)  # 降维

        label_pred = self.svc.predict(data)  # svc分类预测
        accuracy = accuracy_score(self.label, label_pred)
        print("svc预测准确度：", accuracy)
        #print("预测结果：",label_pred)

        del_mask = torch.tensor(label_pred) == 0
        # 删除掩码为1的客户端
        for name in self.user_grad:
            self.user_grad[name] = self.user_grad[name][del_mask]
        #self.grad_list = self.grad_list[del_mask]

    def add_norm(self):
        norms = np.zeros(len(self.user_grad["user_encoder.multihead_attention.W_Q.weight"]))
        print("norm length", len(norms))
        for name in self.user_grad:
            for i in range(len(self.user_grad[name])):
                norm = torch.norm(self.user_grad[name][i]).item()
                norms[i] += norm
        norms = norms.tolist()
        print("最大值位置：", norms.index(max(norms)))
        print("最小值位置：", norms.index(min(norms)))
        norms.sort()
        print("norms: ", norms)
        mean_norm = np.mean(norms)
        norm_coef = mean_norm / norms  # [max(norms) / norm for norm in norms]
        print("norm_coef", norm_coef)
        for name in self.user_grad:
            for i in range(len(self.user_grad[name])):
                self.user_grad[name][i] = self.user_grad[name][i] * norm_coef[i]
        print("归一")

    def update_model_grad(self, all_sample_num, step):
        print("svm agg")

        step_collect = 4
        if step == step_collect:
            self.grad_classify()
        # 预测用户梯度, 归为mal类的客户不参加聚合, 计算归类准确度
        elif step >= step_collect:
            self.grad_pred()
        #self.add_norm()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = torch.sum(
                    self.user_grad[name] / sum(self.user_sample_num.tolist()),
                    dim=0,
                ).cuda()
        self.optimizer.step()

    '''
    ua中的user_grad:由模型名和所有用户参数值的字典组成，{name1:[user[0],user[1]...],name2:...,...,name10:...}
    fl中的grad_list:每个用户的所有参数按顺序堆叠组成[[user[0][name0],...,user[0][name10]],...,[user[-1][name0],...,user[-1][name10]]]'''

    def view_grad(self,mal_list,save_path):
        print("进入可视化")
        #print("mal_list",mal_list)
        # 将mal_list转换为字符串表示的集合
        mal_list_str = set(map(str, mal_list))
        data3 = [grad[11].flatten().numpy() for grad in self.grad_list if str(grad) not in mal_list_str]  # text_encoder.multihead_attention.W_Q.weight
        data3 = np.array(data3)
        data31 = [grad[11].flatten().numpy() for grad in mal_list]
        data31 = np.array(data31)
        data = np.concatenate((data3, data31))

        # 生成三个类别的样本数据的标签
        label1 = np.zeros(data3.shape[0]) + 0

        label11 = np.zeros(data31.shape[0]) + 1

        labels = np.concatenate((label1, label11))
        label_list = ['WQ', 'WQ_mal', 'WK', 'WK_mal', 'WV', 'WV_mal']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(data, labels)
        # 归一化
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)

        for i in range(len(np.unique(labels))):
            plt.scatter(X_norm[labels == i, 0], X_norm[labels == i, 1], s=150, color=colors[i], label=label_list[i], alpha=0.5)
        plt.xlabel('t-SNE Dimension 1', fontsize=20)  # 定义坐标轴标题
        plt.ylabel('t-SNE Dimension 2', fontsize=20)
        plt.title('t-SNE Visualization', fontsize=24)  # 定义图题
        # plt.legend() #图例
        # 调整图例
        plt.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1, 0.5))  # 设置图例字体大小和位置
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        # 保存图像
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()  # 显示图形
