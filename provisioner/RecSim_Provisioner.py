from .HASCO_Provisioner import *
from provisioner.src.utils import *

class RecSimProvisioner(HASCOProvisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'RecSim'
		self.allpowermodels = ['PMB2s', 'PMB2ms', 'PMB4ms', 'PMB8ms']
		costs = np.array([0.08, 0.12, 0.17, 0.33]) / 12
		ipscaps = [2019, 2019, 4029, 16111]
		self.costdict = dict(zip(self.allpowermodels, costs))
		self.ipscaps = dict(zip(self.allpowermodels, ipscaps))
		self.gamma = 0.5 # weight of cost w.r.t utilization ratio

	def updateMetrics(self):
		self.costs = [self.costdict[host.powermodel.__class__.__name__] if host.enable else 0 for host in self.env.hostlist]
		self.util = [(host.ips / host.ipsCap) if host.enable else 0 for host in self.env.hostlist]

	def getReward(self):
		return (np.sum(self.util) - self.gamma * np.sum(self.costs)) * (0 if np.sum(self.util) == 0 else 1)

	def getSimScores(self, host):
		host.enable = True; self.updateMetrics()
		rEnable = self.getReward()
		host.enable = False; self.updateMetrics()
		rDisable = self.getReward()
		return np.array([rDisable, rEnable])

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		cpu = self.host_util
		for i, host in enumerate(self.env.hostlist):
			# generate BNN based recommendation scores
			scores = np.array([self.testall(cpu, prov=i) for i in [0, 1]])
			# generate simulator based recommendation scores
			sim_scores = self.getSimScores(host)
			# run collaborative filtering
			scores += sim_scores
			todo = np.argmax(scores)
			self.env.hostlist[i].enable = True if todo == 1 else False