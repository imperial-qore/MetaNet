import numpy as np
from copy import deepcopy
import random
from .constants import *
from .utils import *

class Opt:
	def __init__(self, ipsdata, env, maxv):
		self.env = env
		self.maxv = maxv
		self.allpowermodels = ['PMB2s', 'PMB4ms', 'PMB8ms']
		costs = np.array([0.08, 0.17, 0.33]) / 12
		ipscaps = [2019, 4029, 16111]
		self.costs = dict(zip(self.allpowermodels, costs))
		self.ipscaps = dict(zip(self.allpowermodels, ipscaps))
		self.ipsdata = ipsdata
		self.decision = {'remove': [], 'add': []}

	def checkdecision(self, decision):
		return not (\
			len(decision['remove']) > remove_limit or \
			len(decision['add']) > add_limit or \
			len(self.env.hostlist) - len(decision['remove']) < 40 or \
			len(self.env.hostlist) - len(decision['remove']) > 60)

	def neighbours(self, decision):
		neighbours = []; numadds = 0
		# add host
		for pmodel in self.allpowermodels:
			dec = deepcopy(decision)
			dec['add'].append(pmodel)
			if self.checkdecision(dec):
				neighbours.append(dec); numadds += 1
		# remove host
		for hostID in range(len(self.env.hostlist)):
			dec = deepcopy(decision)
			if hostID in [i[0] for i in dec['remove']]: continue
			orphaned = self.env.getContainersOfHost(hostID)
			alloc = self.migrateOrphaned(orphaned, hostID, len(dec['add']))
			dec['remove'].append((hostID, alloc))
			if self.checkdecision(dec):
				neighbours.append(dec)
		neighbours.append(deepcopy(decision))
		return neighbours, numadds

	def migrateOrphaned(self, orphaned, inithostid, numnewnodes):
		indices = list(range(len(self.env.hostlist) + numnewnodes))
		indices.remove(inithostid)
		alloc = []
		for o in orphaned:
			random.shuffle(indices)
			for hostID in indices:
				if hostID >= len(self.env.hostlist) or self.env.getPlacementPossible(o, hostID):
					alloc.append((o, hostID))
					break
		return alloc

	def evaluatedecision(self, decision):
		host_alloc = []
		for hostID in range(len(self.env.hostlist)):
			host_alloc.append([])
		for c in self.env.containerlist:
			if c and c.getHostID() != -1: 
				host_alloc[c.getHostID()].append(c.id) 
		new_hids = []; old_hids = list(range(len(host_alloc)))
		# Add new hosts
		for h in decision['add']:
			new_hids.append(len(host_alloc))
			host_alloc.append([])
		new_hids = np.array(new_hids)
		# Migrate orphans
		for dec in decision['remove']:
			for cid, hid in dec[1]:
				host_alloc[hid].append(cid)
		# Remove hosts
		indices = []
		for dec in decision['remove']:
			inithid = dec[0]
			old_hids.remove(inithid)
			indices.append(inithid - len(indices))
			new_hids -= 1
		for i in indices: del host_alloc[i]
		# Balance IPS by migrating to new hosts
		allcaps = [self.env.hostlist[hid].ipsCap for hid in old_hids] + [self.ipscaps[nid] for nid in decision['add']]
		for hid, cids in enumerate(host_alloc):
			if sum([self.ipsdata[cid] for cid in cids]) > allcaps[hid]:
				cid = host_alloc[hid][np.argmax([self.ipsdata[c] for c in host_alloc[hid]])]
				host_alloc[hid].remove(cid)
				fromlist = new_hids if new_hids.shape[0] > 0 else list(range(len(host_alloc))) 
				hid = np.random.choice(fromlist)
				host_alloc[hid].append(cid)
		# Calculate cost
		allpmodels = [host.powermodel.__class__.__name__ for host in self.env.hostlist] + decision['add']
		cost = sum([self.costs[pmodel] for pmodel in allpmodels]) 
		r = sum(self.ipsdata) / sum(allcaps)
		capsr = sum(allcaps) / self.maxv 
		overhead = sum([(1 if 'B2s' in m else 2 if 'B4ms' in m else 4) for m in decision['add']])
		# print(decision, ', Cost', cost, ', R', r, ', IPS', capsr, 'Overhead', overhead)
		return r - 0.5 * cost + 0.4 * capsr - 0.5 * overhead

	def getweights(self, fitness, adds):
		removes = len(fitness) - adds - 1
		weights = np.array([0.5 / (adds+1e-4)] * adds + [0.5 / (removes+1e-4)] * removes + [0.5])
		return weights / np.sum(weights)

class LocalSearch(Opt):
	def __init__(self, ipsdata, env, maxv):
		super().__init__(ipsdata, env, maxv)

	def search(self):
		oldfitness, newfitness = 0, 1
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood, numadds = self.neighbours(self.decision)
			if neighbourhood == []: break
			if np.random.choice([0, 1], p=[0.6, 0.4]): break
			fitness = [self.evaluatedecision(n) for n in neighbourhood]
			index = np.random.choice(list(range(len(fitness))), p=self.getweights(fitness, numadds)) \
				if np.random.random() < 0.3 else np.argmax(fitness)
			self.decision = neighbourhood[index]
			newfitness = fitness[index]
		return self.decision

class ACO(Opt):
	def __init__(self, ipsdata, env, maxv):
		super().__init__(ipsdata, env, maxv)
		self.n = 5

	def search(self):
		oldfitness = [0] * self.n; newfitness = [1] * self.n
		self.decisions = [{'remove': [], 'add': []} for _ in range(self.n)]
		for _ in range(50):
			for ant in range(self.n):
				if newfitness[ant] < oldfitness[ant]: continue
				oldfitness[ant] = newfitness[ant]
				neighbourhood, numadds = self.neighbours(self.decisions[ant])
				if neighbourhood == []: continue
				fitness = [self.evaluatedecision(n) for n in neighbourhood]
				if random.choice([0, 1]): continue
				index = np.random.choice(list(range(len(fitness))), p=self.getweights(fitness, numadds)) \
					if np.random.random() < 0.4 else np.argmax(fitness)
				self.decisions[ant] = neighbourhood[index]
				newfitness[ant] = fitness[index]
		return self.decisions[np.argmax(newfitness)]

class StochasticSearch(Opt):
	def __init__(self, ipsdata, stddata, env, maxv):
		super().__init__(ipsdata, env, maxv)
		self.stddata = np.array(stddata)
		self.k = 0.1
		# Upper Confidence Bound
		self.ipsdata = np.array(self.ipsdata) + self.k * self.stddata

	def search(self):
		oldfitness, newfitness = 0, 1
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood, numadds = self.neighbours(self.decision)
			if np.random.choice([0, 1], p=[0.6, 0.4]): break
			if neighbourhood == []: break
			fitness = [self.evaluatedecision(n) for n in neighbourhood]
			index = np.random.choice(list(range(len(fitness))), p=self.getweights(fitness, numadds)) \
				if np.random.random() < 0.2 else np.argmax(fitness)
			self.decision = neighbourhood[index]
			newfitness = fitness[index]
		return self.decision

class MABSearch(Opt):
	def __init__(self, ipsdata, env, maxv):
		super().__init__(ipsdata, env, maxv)
		self.mab1 = {}
		self.mabT = {}

	def search(self):
		oldfitness, newfitness = 0, 1
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood, numadds = self.neighbours(self.decision)
			if neighbourhood == []: break
			if np.random.choice([0, 1], p=[0.25, 0.75]): break
			fitness = []; weights = []
			for n in neighbourhood:
				f = self.evaluatedecision(n)
				fitness.append(f)
				self.mab1[hashabledict(n)] = self.mab1.get(hashabledict(n), 0) + (1 if f > newfitness else 0)
				self.mabT[hashabledict(n)] = self.mabT.get(hashabledict(n), 0) + 1
				ones, total = 1 + self.mab1[hashabledict(n)], 1 + self.mabT[hashabledict(n)] + self.mab1[hashabledict(n)]
				weights.append(np.random.beta(ones, total))
			weights = np.array(weights); weights /= np.sum(weights)
			# thompson sampling
			index = np.random.choice(list(range(len(fitness))), p=weights) \
				if np.random.random() < 0.6 else np.argmax(fitness)
			self.decision = neighbourhood[index]
			newfitness = fitness[index]
		return self.decision

class CILPSearch(Opt):
	def __init__(self, ipsdata, env, maxv, window_buffer, host_util, model, optimizer, scheduler, loss_list, training):
		super().__init__(ipsdata, env, maxv)
		self.window_buffer = window_buffer
		self.model = model
		self.host_util = host_util
		self.optimizer, self.loss_list, self.scheduler = optimizer, loss_list, scheduler
		# Add latest ips data to window
		self.window_buffer.append(ipsdata)
		temp = np.array(self.window_buffer)
		temp = normalize(temp, 0, self.maxv)
		self.window = convert_to_windows(temp, self.model)[-1]
		feats = self.window.shape[1]
		d = self.window[None, :]
		self.window = d.permute(1, 0, 2)
		self.elem = self.window[-1, :, :].view(1, 1, feats)
		self.training = training

	def cosimulator(self, decision):
		hostips, containerips = deepcopy(self.host_util), self.ipsdata
		old_hids = list(range(len(hostips)))
		hostips += [0 for _ in decision['add']]
		ipscaps = [host.ipsCap for host in self.env.hostlist] + [self.ipscaps[nid] for nid in decision['add']]
		for hid, alloc in decision['remove']:
			if hid in old_hids: 
				old_hids.remove(hid)
				hostips[hid] = 0
			for c, h in alloc:
				hostips[h] += containerips[c]
		r = [hostips[hid] / ipscaps[hid] for hid in range(len(hostips))]
		return r, hostips

	def search(self):
		oldfitness, newfitness = 0, 1
		predold, prednew = 0, 1
		l = nn.BCELoss()
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood, numadds = self.neighbours(self.decision)
			if neighbourhood == []: break
			if np.random.choice([0, 1], p=[0.1, 0.9]): break
			fitness = []; ls = []
			for n in neighbourhood:
				r, newhostips = self.cosimulator(n)
				if 'Attention' in self.model.name:
					predfitness = self.model(self.window)
				else:
					_, predfitness = self.model(self.window, self.elem, torch.tensor([sum(newhostips)]))
				if self.training:
					fitness.append(predfitness.item()); 
					goldfitness = self.evaluatedecision(n)
					gold = torch.DoubleTensor([0.9]) if goldfitness > oldfitness else torch.DoubleTensor([0.1])
					print(gold.item(), predfitness)
					loss = l(predfitness, gold); ls.append(loss.item())
					self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
				else:
					fitness.append(self.evaluatedecision(n)); 
			if self.training:
				self.loss_list.append((np.mean(ls), 0, self.optimizer.param_groups[0]['lr']))
				plot_accuracies(self.loss_list, base_url, self.model, new=True)
				save_model(self.model, self.optimizer, self.scheduler, 0, self.loss_list)
			index = np.random.choice(list(range(len(fitness)))) \
				if np.random.random() < 0.2 else np.argmax(fitness)
			self.decision = neighbourhood[index]
			newfitness = fitness[index]
		return self.decision