import os
from utils.Utils import *
from provisioner.src.utils import load_cpu_dataset, load_scheduler_dataset, load_energy_dataset, plot_accuracies, one_hot
from .models import *
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch
import pickle

base_url = 'scheduler/src/'
num_epochs = 10

def backprop(epoch, model, optimizer, scheduler, data_cpu, data_scheduler, data_energy, training = True):
	feats = data_cpu.shape[1]; ls = []
	l = nn.MSELoss(reduction = 'mean'); l2 = nn.BCELoss()
	for i, d in enumerate(data_cpu):
		gold = data_energy[i]
		if 'GOBI' in model.name:
			apps, scheds = data_scheduler[i]
			preds = torch.stack([model(d, apps[j], scheds[j]) for j in range(len(apps))])
			loss = l(preds, gold)
		if 'GOSH' in model.name:
			apps, scheds = data_scheduler[i]
			loss = 0
			for j in range(len(apps)):
				pred = model(d, apps[j], scheds[j])
				loss += KL_loss(pred, gold)
		ls.append(torch.mean(loss).item())
		if training:
			optimizer.zero_grad(); loss.backward(); optimizer.step()
	if training: scheduler.step()
	return np.mean(ls), optimizer.param_groups[0]['lr']

def save_model(model, optimizer, scheduler, epoch, loss_list):
	folder = base_url + f'checkpoints/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/{model.name}.ckpt'
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'loss_list': loss_list}, file_path)

def load_model(modelname, dims):
	import scheduler.src.models
	model_class = getattr(scheduler.src.models, modelname)
	model = model_class(dims)
	optimizer = torch.optim.Adam(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	folder = base_url + f'checkpoints/'
	fname = f'{folder}/{model.name}.ckpt'
	if os.path.exists(fname):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		loss_list = checkpoint['loss_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; loss_list = []
	return model, optimizer, scheduler, epoch, loss_list

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

def scale(data, low):
    return torch.max(torch.tensor(low), torch.min(data, torch.tensor(1)))

def gobi_opt(model, cpu, app, init):
    optimizer = torch.optim.AdamW([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100
    while iteration < 200:
        alloc_old = deepcopy(init.data)
        z = model(cpu, app, init)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = scale(init.data, 0)
        equal = equal + 1 if torch.all(alloc_old - init < 0.01) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    init.requires_grad = False 
    return init