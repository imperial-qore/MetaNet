from .HASCO_Scheduler import *
from provisioner.src.utils import *
from .src.opt import *

class RecSimScheduler(HASCOScheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'RecSim'
		self.alpha, self.beta, self.delta = 0.3, 0.3, 0.3
		self.rt_dict, self.sla_dict = get_dicts()
		self.estimateTaskIPS = get_estimate_ips(); 

	def getSimScores(self, task):
		sim_scores = []; self.emax = self.env.provisioner.emax
		for hostID in range(len(self.env.hostlist)):
			ehosts = []; 
			application, choice = task.application, task.choice
			for host in self.env.hostlist:
				if host.id == hostID:
					ehosts.append(host.getPowerFromIPS(host.ips + self.estimateTaskIPS))
				else:
					ehosts.append(host.getPower())
			e = np.sum(ehosts)
			e = e / self.emax # normalize
			sla = self.sla_dict[application]
			rt = np.random.normal(loc = self.rt_dict[application][choice][0], scale = self.rt_dict[application][choice][1])
			if self.env.hostlist[hostID].getIPSAvailable() > self.estimateTaskIPS:
				rt /= 2
			sla_v = (rt <= sla) + 0
			sim_scores.append(1 - (self.alpha * e + self.delta * sla_v))
		return sim_scores

	def placement(self, tasks):
		if not self.model_loaded: self.load_model()
		start = time()
		cpu = self.host_util
		decisions = []
		for task in tasks:
			# generate BNN based recommendation scores
			scores = np.array([self.testall(cpu, prov=1, app=task.application, dec=task.choice, sched=i) for i in list(range(len(self.env.hostlist)))])
			# generate simulator based recommendation scores
			sim_scores = self.getSimScores(task)
			# run collaborative filtering
			scores += sim_scores
			hostID = np.argmax(scores)
			decisions.append(hostID)
		return decisions, time() - start