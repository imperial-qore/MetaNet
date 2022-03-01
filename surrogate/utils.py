import pandas as pd
import numpy as np
from .model import *

DSET_PATH = 'data/'
NUM_EPOCHS = 10

def normalize(x):
    range_x = (torch.min(x), torch.max(x))
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return x, range_x

def trainModel(N_HOSTS, schedulers):
    model = Surrogate(N_HOSTS, len(schedulers)).double()
    optimizer = torch.optim.Adam(model.parameters() , lr=model.lr, weight_decay=1e-5)
    cpu_dfs, rt_dfs, sched_name = [], [], []
    for scheduler in schedulers:
        cpu_dfs.append(pd.read_csv(DSET_PATH + scheduler + '/cpu_with_interval.csv', header=None))
        rt_dfs.append(pd.read_csv(DSET_PATH + scheduler + '/avgresponsetime_with_interval.csv', header=None))
        sched_name.extend([scheduler] * len(cpu_dfs[-1]))
    dset_cpu = torch.tensor(pd.concat(cpu_dfs).values)
    dset_rt =  torch.tensor(pd.concat(rt_dfs).values)
    dset_rt, range_rt = normalize(dset_rt)

    l = nn.MSELoss(reduction = 'mean')
    for epoch in range(NUM_EPOCHS):
        ls = []
        for i in range(dset_cpu.shape[0]):
            index = schedulers.index(sched_name[i])
            pred = model(dset_cpu[i])[index]
            topred = dset_rt[i]
            loss = l(pred, topred)
            ls.append(loss.item())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f'Epoch {epoch}, Loss = {np.mean(ls)}')
    return model

