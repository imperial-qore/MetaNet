import numpy as np
from copy import deepcopy
import random
from .constants import *
from .utils import *

class Opt:
	def __init__(self, ipsdata, env):
		self.env = env
		self.allpowermodels = ['PMB2s', 'PMB4ms', 'PMB8ms']
		costs = np.array([0.08, 0.17, 0.33]) / 12
		ipscaps = [2019, 4029, 16111]
		self.costs = dict(zip(self.allpowermodels, costs))
		self.ipscaps = dict(zip(self.allpowermodels, ipscaps))
		self.ipsdata = ipsdata
		self.decision = [1 for _ in range(len(self.env.hostlist))]

	def checkdecision(self, decision):
		return not (np.sum(decision) < 0.5 * len(self.env.hostlist))

	def neighbours(self, decision):
		neighbours = []
		for hostID in range(len(self.env.hostlist)):
			dec = deepcopy(decision)
			dec[hostID] = 1 if decision[hostID] == 0 else 0
			if self.checkdecision(dec):
				neighbours.append(dec)
		neighbours.append(deepcopy(decision))
		return neighbours

	def evaluatedecision(self, decision):
		allcaps = [host.ipsCap for host in self.env.hostlist]
		allpmodels = [host.powermodel.__class__.__name__ for host in self.env.hostlist]
		cost = sum([self.costs[host.powermodel.__class__.__name__] for hostID, host in enumerate(self.env.hostlist) if decision[hostID]]) 
		r = sum(self.ipsdata)  / sum(allcaps)
		return r - 0.2 * cost

class LocalSearch(Opt):
	def __init__(self, ipsdata, env):
		super().__init__(ipsdata, env)

	def search(self):
		oldfitness, newfitness = 0, 1
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood = self.neighbours(self.decision)
			if neighbourhood == []: break
			if np.random.choice([0, 1], p=[0.6, 0.4]): break
			fitness = [self.evaluatedecision(n) for n in neighbourhood]
			index = np.random.choice(list(range(len(fitness)))) \
				if np.random.random() < 0.3 else np.argmax(fitness)
			self.decision = neighbourhood[index]
			newfitness = fitness[index]
		return self.decision

class ACO(Opt):
	def __init__(self, ipsdata, env):
		super().__init__(ipsdata, env)
		self.n = 5

	def search(self):
		oldfitness = [0] * self.n; newfitness = [1] * self.n
		self.decisions = [[1] * len(self.env.hostlist) for _ in range(self.n)]
		for _ in range(50):
			for ant in range(self.n):
				if newfitness[ant] < oldfitness[ant]: continue
				oldfitness[ant] = newfitness[ant]
				neighbourhood = self.neighbours(self.decisions[ant])
				if neighbourhood == []: continue
				fitness = [self.evaluatedecision(n) for n in neighbourhood]
				if random.choice([0, 1]): continue
				index = np.random.choice(list(range(len(fitness)))) \
					if np.random.random() < 0.4 else np.argmax(fitness)
				self.decisions[ant] = neighbourhood[index]
				newfitness[ant] = fitness[index]
		return self.decisions[np.argmax(newfitness)]

class StochasticSearch(Opt):
	def __init__(self, ipsdata, stddata, env):
		super().__init__(ipsdata, env)
		self.stddata = np.array(stddata)
		self.k = 0.1
		# Upper Confidence Bound
		self.ipsdata = np.array(self.ipsdata) + self.k * self.stddata

	def search(self):
		oldfitness, newfitness = 0, 1
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood = self.neighbours(self.decision)
			if np.random.choice([0, 1], p=[0.6, 0.4]): break
			if neighbourhood == []: break
			fitness = [self.evaluatedecision(n) for n in neighbourhood]
			index = np.random.choice(list(range(len(fitness)))) \
				if np.random.random() < 0.2 else np.argmax(fitness)
			self.decision = neighbourhood[index]
			newfitness = fitness[index]
		return self.decision

class MABSearch(Opt):
	def __init__(self, ipsdata, env):
		super().__init__(ipsdata, env)
		self.mab1 = {}
		self.mabT = {}

	def search(self):
		oldfitness, newfitness = 0, 1
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood = self.neighbours(self.decision)
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