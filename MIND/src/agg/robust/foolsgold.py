import torch
from torch import nn

from agg.user import UserAggregator

class FoolsGold(UserAggregator):
    def foolsgold(self):
        print("进入聚合")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"参数 '{name}' 在聚合前出现 NaN 值")
                break

        param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in self.grad_list])
        num_workers = len(param_list)
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to()
        cs = torch.zeros((num_workers, num_workers)).to(self.device)
        for i in range(num_workers):
            for j in range(i):
                ## compute cosine similarity
                cs[i, j] = cos(param_list[i], param_list[j])
                cs[j, i] = cs[i, j]
        ###The foolsgold algorithm implemented below
        v = torch.zeros(num_workers).to(self.device)
        for i in range(num_workers):
            v[i] = torch.max(cs[i])

        alpha = torch.zeros(num_workers).to(self.device)
        for i in range(num_workers):
            for j in range(num_workers):
                if (v[j] > v[i]):
                    cs[i, j] = cs[i, j] * v[i] / v[j]
            alpha[i] = 1 - torch.max(cs[i])

        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha = alpha / (torch.max(alpha))
        alpha[alpha == 1] = 0.99
        alpha = torch.log(alpha / (1 - alpha)) + 0.5
        alpha[(torch.isinf(alpha) + (alpha > 1))] = 1
        alpha[alpha < 0] = 0
        alpha = alpha / (torch.sum(alpha).item() + 1e-8)  # 添加小正则化项以避免除数为0

        if torch.isnan(alpha).any():
            print("Normalized alpha contains NaN.")

        param_list = param_list.to(self.device)
        alpha = alpha.to(self.device)
        global_params = torch.matmul(torch.transpose(param_list, 0, 1), alpha.reshape(-1, 1))
        del param_list
        global_params = global_params.to(self.device)
        return global_params
        # print(time.time()-start)


    def update_model_grad(self,all_sample_num,step,root_user_index):
        global_params = self.foolsgold()
        idx = 0
        for j, (param) in enumerate(self.model.named_parameters()):
            if param[1].requires_grad:
                param[1].grad = global_params[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()
        self.optimizer.step()
        del global_params
        print("聚合结束")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"参数 '{name}' 在聚合后出现 NaN 值")
                break
            else:
                print("聚合后不为nan")
                break
