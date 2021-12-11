import numpy as np
from copy import deepcopy
import random
from .utils import *

class Opt:
	def __init__(self, ipsdata, env, applist, slalist):
		self.env = env
		self.choices = ['layer', 'semantic', 'compression']
		self.rt_dict, self.sla_dict = get_dicts()
		self.a_dict = get_accuracies()
		self.applist, self.slalist = applist, slalist
		self.ipsdata = ipsdata
		self.decision = random.choices(self.choices, k=len(self.applist))

	def neighbours(self, decision):
		neighbours = []
		for appid in range(len(self.applist)):
			for choice in self.choices:
				if choice == decision[appid]: continue
				dec = deepcopy(decision)
				dec[appid] = choice
				neighbours.append(dec)
		neighbours.append(deepcopy(decision))
		return neighbours

	def evaluatedecision(self, decision):
		sla_estimates = [self.sla_dict[application] for application in self.applist]
		rt_estimates = [np.random.normal(loc = self.rt_dict[self.applist[i]][decision[i]][0], scale = self.rt_dict[self.applist[i]][decision[i]][1]) for i in range(len(decision))]
		sla_scores = [rt_estimates[i] <= sla_estimates[i] for i in range(len(decision))]
		acc_scores = [self.a_dict[self.applist[i]][decision[i]] for i in range(len(decision))]
		scores = [sla_scores[i] + acc_scores[i] for i in range(len(decision))]
		return np.mean(scores)

class LocalSearch(Opt):
	def __init__(self, ipsdata, env, applist, slalist):
		super().__init__(ipsdata, env, applist, slalist)

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
	def __init__(self, ipsdata, env, applist, slalist):
		super().__init__(ipsdata, env, applist, slalist)
		self.n = 5

	def search(self):
		oldfitness = [0] * self.n; newfitness = [1] * self.n
		self.decisions = [random.choices(self.choices, k=len(self.applist)) for _ in range(self.n)]
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
	def __init__(self, ipsdata, stddata, env, applist, slalist):
		super().__init__(ipsdata, env, applist, slalist)
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
	def __init__(self, ipsdata, env, applist, slalist):
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