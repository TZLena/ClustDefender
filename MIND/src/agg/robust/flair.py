import torch
from torch import nn

from agg.user import UserAggregator

class FLAIR(UserAggregator):

    def flair(self):
        print("flair agg")

        # reshaping the parameter list
        cmax = 2
        param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in self.grad_list])
        param_list = param_list.to(self.device)
        for param in self.susp:
            if(param==None): print("susp处理前有nan")
        for param in param_list:
            if(param==None): print("param list有nan")
        '''# FS_min and FS_max used by an adversary in an adaptive attack
        fs_min = torch.sort(fs)[0][cmax - 1]
        fs_max = torch.sort(fs)[0][-cmax]
        if 'adaptive_krum' in str(byz):  # if the attack is adaptive
            param_list = byz(device, lr, param_list, old_direction, cmax, fs_min, fs_max)
        elif 'adaptive_trim' in str(byz):
            param_list = byz(device, lr, param_list, old_direction, cmax, fs_min, fs_max, weight)
        else:
            param_list = byz(device, lr, param_list, cmax)  # non-adaptive attack'''

        flip_local = torch.zeros(len(param_list))  # flip-score vector
        penalty = 1.0 - 2 * cmax / len(param_list)
        reward = 1.0 - penalty

        ##Computing flip-score
        for i in range(len(param_list)):
            direction = torch.sign(param_list[i]).to(self.device)
            flip = torch.sign(direction * (direction - self.direction.reshape(-1)))
            flip_local[i] = torch.sum(flip * (param_list[i] ** 2))
            if torch.isnan(flip_local).any():
                print("flip_local处理后有nan")
            del direction, flip

        # updating self.suspicion-score
        argsorted = torch.argsort(flip_local).to(self.device)
        if (cmax > 0):
            self.susp[argsorted[cmax:-cmax]] = self.susp[argsorted[cmax:-cmax]] + reward
            self.susp[argsorted[:cmax]] = self.susp[argsorted[:cmax]] - penalty
            self.susp[argsorted[-cmax:]] = self.susp[argsorted[-cmax:]] - penalty
        argsorted = torch.argsort(self.susp)
        print("处理后的susp",self.susp)
        if torch.isnan(self.susp).any():
            print("susp处理后有nan")
        if torch.sum(torch.exp(self.susp)) == 0 :
            print("weights分母为0，结果为nan")
        # updating weights
        weights = torch.exp(self.susp) / (torch.sum(torch.exp(self.susp)) + 1e-10)

        global_params = torch.matmul(torch.transpose(param_list, 0, 1), weights.reshape(-1, 1))
        self.direction = torch.sign(global_params)

        return global_params


    def update_model_grad(self,all_sample_num,step, root_user_index):
        global_params = self.flair()
        idx = 0
        for j, (param) in enumerate(self.model.named_parameters()):
            if param[1].requires_grad:
                param[1].grad = global_params[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()

        self.optimizer.step()
        print("聚合结束")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"参数 '{name}' 在fltrust聚合后出现 NaN 值")
            else:
                print("聚合后不为nan")
                break
