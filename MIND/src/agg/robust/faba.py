import numpy as np
import torch
from torch import nn

from agg.user import UserAggregator


class FABA(UserAggregator):
    def faba(self):
        print("进入聚合")
        cmax = 2
        param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in self.grad_list])
        param_list = param_list.to(self.device)
        #param_list = byz(device, lr, param_list, cmax)
        faba_client_list = np.ones(len(param_list))  # contains the current benign clients
        dist = np.zeros(len(param_list))
        G0 = torch.mean(param_list, dim=0).to(self.device)
        for i in range(cmax):
            for j in range(len(param_list)):
                if faba_client_list[j]:
                    dist[j] = torch.norm(G0 - param_list[j]).item()
            outlier = int(np.argmax(dist))
            faba_client_list[outlier] = 0  # outlier removed as suspected
            dist[outlier] = 0
            G0 = (G0 * (len(param_list) - i) - param_list[outlier]) / (len(param_list) - i - 1)  # mean recomputed

        del param_list
        # print(time.time()-start)
        return G0

    def update_model_grad(self, all_sample_num,step,root_user_index):
        G0 = self.faba()
        idx = 0
        for j, (param) in enumerate(self.model.named_parameters()):
            if param[1].requires_grad:
                param[1].grad = G0[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()

        self.optimizer.step()
