import pickle
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
from csv import reader
from utils.Utils import *
from .models import *

base_url = 'scheduler/src/'

def one_hot(value, lst):
	vec = torch.zeros(len(lst))
	vec[lst.index(value)] = 1
	return vec

def load_cpu_dataset(feats):
	fname = base_url + f'datasets/cpu_with_interval.csv'
	dset = np.abs(np.genfromtxt(fname, delimiter=',')).reshape(-1, feats) / 100
	dset  = torch.FloatTensor(dset)
	split = int(0.9 * dset.shape[0])
	train, test = dset[:split], dset[split:]
	return train, test

def load_provisioner_dataset(feats):
	fname = base_url + f'datasets/cpu_with_interval.csv'
	dset_cpu = np.abs(np.genfromtxt(fname, delimiter=',')).reshape(-1, feats) / 100
	fname = base_url + f'datasets/enable_with_interval.csv'
	dset = []
	with open(fname, 'r') as read_obj:
		csv_reader = reader(read_obj); row_num = 0
		for row in csv_reader:
			enable = [one_hot(int(cpu), [0, 1]) for cpu in row]
			dset.append((torch.FloatTensor(dset_cpu[row_num]), enable))
			row_num += 1
	split = int(0.9 * len(dset))
	train, test = dset[:split], dset[split:]
	return train, test

def load_decider_dataset(feats):
	fname = base_url + f'datasets/decider_with_interval.csv'
	fn_names = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]
	choices = ['layer', 'semantic', 'compression']
	dset = []
	with open(fname, 'r') as read_obj:
		csv_reader = reader(read_obj)
		for row in csv_reader:
			lst = [i for i in row if i]
			apps = [one_hot(i, fn_names) for i in lst[:len(lst)//2]]
			cs = [one_hot(i, choices) for i in lst[len(lst)//2:]]
			dset.append((apps, cs))
	split = int(0.9 * len(dset))
	train, test = dset[:split], dset[split:]
	return train, test

def load_scheduler_dataset(feats):
	fname = base_url + f'datasets/scheduler_with_interval.csv'
	fn_names = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]
	dset = []
	with open(fname, 'r') as read_obj:
		csv_reader = reader(read_obj)
		for row in csv_reader:
			lst = [i for i in row if i]
			apps = [one_hot(i, fn_names) for i in lst[:len(lst)//2]]
			schedule = [one_hot(float(i)//1, list(range(feats))) for i in lst[len(lst)//2:]]
			dset.append((apps, schedule))
	split = int(0.9 * len(dset))
	train, test = dset[:split], dset[split:]
	return train, test

def load_energy_dataset(feats):
	fname = base_url + f'datasets/energy_with_interval.csv'
	dset = np.abs(np.genfromtxt(fname, delimiter=',')).reshape(-1, 1)
	maxe = np.max(dset)
	dset  = torch.FloatTensor(dset) / maxe
	split = int(0.9 * dset.shape[0])
	train, test = dset[:split], dset[split:]
	return train, test, maxe

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
	import provisioner.src.models
	model_class = getattr(provisioner.src.models, modelname)
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

def backprop(epoch, model, optimizer, scheduler, data_cpu, data_provisioner, data_decider, data_scheduler, training = True, data_energy = None):
	feats = data_cpu.shape[1]; ls = []
	l = nn.MSELoss(reduction = 'mean'); l2 = nn.BCELoss()
	for i, d in enumerate(data_cpu):
		gold = data_cpu[i+1][-1] if i+1 < data_cpu.shape[0] else data_cpu[i][-1]
		if 'LSTM' in model.name:
			pred = model(d)
			loss = l(pred, gold)
		elif 'Attention' in model.name:
			pred = model(d)
			loss = l(pred, gold)
		elif 'GNN' in model.name:
			pred = model(d)
			loss = l(pred, gold)
		elif 'Transformer' in model.name:
			d = d[None, :]
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, 1, feats)
			pred = model(window, elem).view(-1)
			loss = l(pred, gold)
		elif 'NPN' in model.name:
			pred = model(d)
			loss = KL_loss(pred, gold)
		elif 'HASCO' in model.name or 'RecSim' in model.name:
			_, p_in = data_provisioner[i]
			app_in, d_in = data_decider[i]
			_, s_in = data_scheduler[i]
			preds = []
			for p in p_in:
				ps = []
				for i in range(len(app_in)):
					app, dec, sched = app_in[i], d_in[i], s_in[i]
					ps.append(model(d, p, app, dec, sched))
				preds.append(torch.stack(ps))
			loss = l(torch.stack(preds), data_energy[i])
		elif 'Sci' in model.name:
			# Window prediction
			window = d.view(1, 1, feats)
			elem = window[-1, :, :].view(1, 1, feats)
			memory, pred = model.predwindow(window, elem)
			pred = pred.view(-1)
			loss = l(pred, gold)
			# CILP Imitation Learning
			p_in, p_gold = data_provisioner[i]
			p_out = torch.stack([model.forward_provisioner(memory, i) for i in p_in])
			d_in, d_gold = data_decider[i]
			d_out = torch.stack([model.forward_decider(memory, i, p_out) for i in d_in])
			s_in, s_gold = data_scheduler[i]
			s_out = torch.stack([model.forward_scheduler(memory, i, p_out) for i in d_in])
			p_gold, d_gold, s_gold = torch.stack(p_gold), torch.stack(d_gold), torch.stack(s_gold)
			loss = loss + l2(p_out, p_gold) + l2(d_out, d_gold) + l2(s_out, s_gold)
		ls.append(torch.mean(loss).item())
		if training:
			optimizer.zero_grad(); loss.backward(); optimizer.step()
	if training: scheduler.step()
	return np.mean(ls), optimizer.param_groups[0]['lr']

def plot_accuracies(loss_list, folder, model, new=False):
	os.makedirs(f'{folder}/plots/', exist_ok=True)
	trainAcc = [i[0] for i in loss_list]
	testAcc = [i[1] for i in loss_list]
	lrs = [i[1] for i in loss_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', color='k', linewidth=1, linestyle='solid', marker='.')
	plt.plot(range(len(testAcc)), testAcc, label='Average Testing Loss', color='b', linewidth=1, linestyle='dashed', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='dotted', marker='.')
	cnew = '_online' if new else ''
	plt.legend()
	plt.savefig(f'{folder}/plots/{model.name}{cnew}.pdf')
	plt.clf()

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

def hashabledict(dict):
  return json.dumps(dict)