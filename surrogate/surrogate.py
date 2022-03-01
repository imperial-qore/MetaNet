import pandas as pd
import numpy as np
from .model import *
from copy import deepcopy
from scheduler.src.gobi import *

FEATS = 9
DSET_PATH = 'data/sim_surrogate/'
REAL_PATH = 'data/real/'
NUM_EPOCHS = 10

def train_scheduler():
    self.feats = HOSTS
    self.model, self.optimizer, self.scheduler, epoch, self.loss_list = load_model(self.model_name, self.feats)
    # Load dataset
    train_cpu, test_cpu = load_cpu_dataset(self.feats)
    train_scheduler, test_scheduler = load_scheduler_dataset(self.feats)
    train_energy, test_energy, self.emax = load_energy_dataset(self.feats)
    # Train model 
    if epoch == -1:
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr = backprop(e, self.model, self.optimizer, self.scheduler, train_cpu, train_scheduler, train_energy)
            lossTest, _ = backprop(e, self.model, self.optimizer, self.scheduler, test_cpu, test_scheduler, train_energy, False)
            self.loss_list.append((lossT, lossTest, lr))
            tqdm.write(f'Epoch {e},\tTrain Loss = {lossT},\tTest loss = {lossTest}')
            plot_accuracies(self.loss_list, base_url, self.model)
        save_model(self.model, self.optimizer, self.scheduler, e, self.loss_list)

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

def normalize(x):
    range_x = (torch.min(x), torch.max(x))
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return x, range_x

def trainModel(N_HOSTS, N_FEATS):
    model = Surrogate(N_HOSTS, N_FEATS).double()
    optimizer = torch.optim.Adam(model.parameters() , lr=model.lr, weight_decay=1e-5)
    dset_cpu = torch.tensor(pd.read_csv(DSET_PATH + 'cpu_with_interval.csv', header=None).values)
    dset_params =  torch.tensor(pd.read_csv(DSET_PATH + 'params_with_interval.csv', header=None).values)
    dset_en =  torch.tensor(pd.read_csv(DSET_PATH + 'energytotalinterval_with_interval.csv', header=None).values)
    dset_rt =  torch.tensor(pd.read_csv(DSET_PATH + 'avgresponsetime_with_interval.csv', header=None).values)
    # dset_en, range_en = normalize(dset_en); dset_rt, range_rt = normalize(dset_rt)

    l = nn.MSELoss(reduction = 'mean')
    for epoch in range(NUM_EPOCHS):
        ls = []
        for i in range(dset_cpu.shape[0]):
            cpu, params = dset_cpu[i], dset_params[i]
            pred = model(cpu.view(1, -1), params.view(1, -1)).view(-1)
            topred = torch.cat((dset_en[i], dset_rt[i]))
            loss = l(pred, topred)
            ls.append(loss.item())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f'Epoch {epoch}, Loss = {np.mean(ls)}')
    freeze(model)
    return model

def scale(data, low):
    return torch.max(torch.tensor(low), data)

def opt(model, N_HOSTS):
    # initial simulator params
    dset_params = pd.read_csv(DSET_PATH + 'params_with_interval.csv', header=None).values
    init = torch.tensor(dset_params[0], dtype=torch.float, requires_grad=True)

    # tune simulator params
    dset_cpu = torch.tensor(pd.read_csv(REAL_PATH + 'cpu_with_interval.csv', header=None).values)
    dset_en =  torch.tensor(pd.read_csv(REAL_PATH + 'energytotalinterval_with_interval.csv', header=None).values).view(-1, 1)
    dset_rt =  torch.tensor(pd.read_csv(REAL_PATH + 'avgresponsetime_with_interval.csv', header=None).values).view(-1, 1)
    topred = torch.cat((dset_en, dset_rt), dim=1)

    l = nn.MSELoss(reduction = 'mean')
    optimizer = torch.optim.AdamW([init] , lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100
    while iteration < 600:
        param_old = deepcopy(init.data)
        pred = model(dset_cpu, init.repeat(dset_cpu.shape[0], 1))
        z = l(pred, topred)
        if iteration % 50 == 0: print('Surrogate Loss with Real Trace', z.item())
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = scale(init.data, 0)
        equal = equal + 1 if torch.all(param_old - init < 0.01) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    init.requires_grad = False 
    return init.view(N_HOSTS, -1).tolist()

def load_cpu_dataset(feats):
	fname = f'data/sim/cpu_with_interval.csv'
	dset = np.abs(np.genfromtxt(fname, delimiter=',')).reshape(-1, feats) / 100
	dset  = torch.FloatTensor(dset)
	split = int(0.9 * dset.shape[0])
	train, test = dset[:split], dset[split:]
	return train, test

def train_scheduler(HOSTS):
    # Load model
    model, optimizer, scheduler, epoch, loss_list = load_model('GOBI', HOSTS)
    # Load dataset
    train_cpu, test_cpu = load_cpu_dataset(HOSTS)
    train_scheduler, test_scheduler = load_scheduler_dataset(HOSTS)
    train_energy, test_energy, emax = load_energy_dataset(HOSTS)
    # Train model 
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
        lossT, lr = backprop(e, model, optimizer, scheduler, train_cpu, train_scheduler, train_energy)
        lossTest, _ = backprop(e, model, optimizer, scheduler, test_cpu, test_scheduler, train_energy, False)
        loss_list.append((lossT, lossTest, lr))
        tqdm.write(f'Epoch {e},\tTrain Loss = {lossT},\tTest loss = {lossTest}')
    # Freeze encoder
    freeze(model)
    return model
