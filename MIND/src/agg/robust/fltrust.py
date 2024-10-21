import torch
from torch import nn

from agg.user import UserAggregator

class FLTrust(UserAggregator):
    def fltrust(self, root_user_index):  # grad_list:嵌套列表，由所有用户、每个用户的所有参数组成，不含名称
        print("进入聚合")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"参数 '{name}' 在fltrust聚合前出现 NaN 值")
                break
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        #param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in self.grad_list])
        flattened_gradients = []

        for x in self.grad_list:
            flattened_params = []
            for xx in x:
                # print("grad in one user:",xx.shape)
                flattened_param = xx.reshape((-1))  # 展平每个用户的所有参数
                # print("grad after flattened:", flattened_param.shape)
                flattened_params.append(flattened_param)
            concatenated_params = torch.cat(flattened_params, dim=0)  # 连接成一个一维张量
            # print("grad after cat:", concatenated_params.shape)
            flattened_gradients.append(concatenated_params)
        param_list = torch.stack(flattened_gradients, dim=0)  # 堆叠所有用户的梯度形成二维张量
        # Client -1 acts as the root dataset holder
        server_params = param_list[root_user_index]  # 服务器算出的更新 在抽取投毒客户时最后一个客户端永远不会被选中
        # print("server_params",server_params)
        server_norm = torch.norm(server_params)
        if root_user_index == 0:
            param_list = param_list[1:]
        else:
            param_list = param_list[:root_user_index] + param_list[root_user_index + 1:]  # [np.random.permutation(tau)]

        '''server_params = param_list[-1]  # 服务器算出的更新 此处改为最后一个 在抽取投毒客户时最后一个客户端永远不会被选中
        # print("server_params",server_params)
        server_norm = torch.norm(server_params)
        param_list = (param_list[:-1])  # [np.random.permutation(tau)]'''
        # print("param_list", param_list.shape)
        #param_list = byz(device, lr, param_list, nbyz) # 经过投毒训练后的梯度

        mal_cos = []
        beg_cos = []
        cossim = []
        for i in range(len(param_list)):
            sim = cos(server_params, param_list[i]).item()
            '''if i in uindex:
                mal_cos.append(sim)
            else:
                beg_cos.append(sim)'''
            cossim.append(sim)
        print("相似度大于90的数量：",sum(1 for x in cossim if x < 0))
        # Combine and sort both lists
        '''combined_cos = mal_cos + beg_cos
        sorted_cos = sorted(combined_cos)

        # Count the number of values less than 0
        num_less_than_zero = sum(1 for x in sorted_cos if x < 0)

        # Find the ranks of mal_cos values in the sorted list
        mal_cos_ranks = [sorted_cos.index(x) + 1 for x in mal_cos]

        # Sort mal_cos and beg_cos individually
        sorted_mal_cos = sorted(mal_cos)
        sorted_beg_cos = sorted(beg_cos)

        #print("Sorted mal_cos:", sorted_mal_cos)
        #print("Sorted beg_cos:", sorted_beg_cos)
        print("Number of values less than 0:", num_less_than_zero)
        #print("Ranks of mal_cos values in the combined sorted list:", mal_cos_ranks)'''

        # The FLTRUST algorithm
        ts = torch.zeros((len(param_list)))  # 信誉分数初始化
        for i in range(len(param_list)): # 聚合时排除服务器
            ts[i] = max(cos(server_params, param_list[i]), 0)  # 算出每个用户的信誉分数，小于零排除
            param_list[i] = (server_norm / torch.norm(param_list[i])) * param_list[i] * ts[i]  # 归一化
            param_norm = torch.norm(param_list[i])
        if torch.sum(ts) == 0:
            print("Sum of trust scores is zero.")
            global_params = server_params.clone()
        else:
            global_params = torch.sum(param_list, dim=0) / torch.sum(ts)  # 信誉分数加权聚合
        global_params = global_params.to(self.device)
        return global_params


    def update_model_grad(self,all_sample_num,step, root_user_index):
        global_params = self.fltrust(root_user_index)
        if torch.isnan(global_params).any():
            print("Global params contain NaN.")
        if torch.isinf(global_params).any():
            print("Global params contain Inf.")
        idx = 0
        for name, param in self.model.named_parameters():
            # print(j,"param before agg",param[0],param[1].data)
            if param.requires_grad:
                param.grad = global_params[idx:(idx + param.nelement())].reshape(param.shape).cuda()
                idx += param.nelement()
        self.optimizer.step()
        print("聚合结束")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"参数 '{name}' 在fltrust聚合后出现 NaN 值")
                break
            else:
                print("聚合后不为nan")
                break
