from .Scheduler import *
from .src.decider_utils import *
import numpy as np
from copy import deepcopy

class CoSimScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.alpha, self.beta, self.delta = 0.3, 0.3, 0.3
		self.rt_dict, self.sla_dict = get_dicts()
		self.emax = 0
		self.estimateTaskIPS = get_estimate_ips(); 
		print('Estimate Task IPS:', self.estimateTaskIPS)

	def runSimpleSimulation(self, task, hostID):
		ehosts = []; 
		application, choice = task.application, task.choice
		for host in self.env.hostlist:
			if host.id == hostID:
				ehosts.append(host.getPowerFromIPS(host.ips + self.estimateTaskIPS))
			else:
				ehosts.append(host.getPower())
		e = np.sum(ehosts)
		self.emax = max(e, self.emax); e = e / self.emax # normalize
		sla = self.sla_dict[application]
		rt = np.random.normal(loc = self.rt_dict[application][choice][0], scale = self.rt_dict[application][choice][1])
		if self.env.hostlist[hostID].getIPSAvailable() > self.estimateTaskIPS:
			rt /= 2
		sla_v = (rt <= sla) + 0
		return 1 - (self.alpha * e + self.delta * sla_v)

	def placement(self, tasks):
		start = time(); decision = []
		for task in tasks:
			scores = [self.runSimpleSimulation(task, hostID) for hostID, _ in enumerate(self.env.hostlist)]
			for hostID, host in enumerate(self.env.hostlist): # do not schedule on disabled hosts
				if not host.enable: scores[hostID] = -1000
			decision.append(np.argmax(scores))
		return decision, time() - start