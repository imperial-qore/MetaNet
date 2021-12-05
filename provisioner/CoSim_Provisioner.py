from .Provisioner import *

class CoSimProvisioner(Provisioner):
	def __init__(self):
		super().__init__()
		self.allpowermodels = ['PMB2s', 'PMB4ms', 'PMB8ms']
		costs = np.array([0.08, 0.17, 0.33]) / 12
		ipscaps = [2019, 4029, 16111]
		self.costdict = dict(zip(self.allpowermodels, costs))
		self.ipscaps = dict(zip(self.allpowermodels, ipscaps))
		self.gamma = 0.5 # weight of cost w.r.t utilization ratio

	def updateMetrics(self):
		self.costs = [self.costdict[host.powermodel.__class__.__name__] if host.enable else 0 for host in self.env.hostlist]
		self.util = [(host.ips / host.ipsCap) if host.enable else 0 for host in self.env.hostlist]

	def getReward(self):
		return (np.sum(self.util) - self.gamma * np.sum(self.costs)) * (0 if np.sum(self.util) == 0 else 1)

	def provision(self):
		for host in self.env.hostlist:
			# run A/B testing
			host.enable = True
			self.updateMetrics()
			rEnable = self.getReward()
			host.enable = False
			self.updateMetrics()
			rDisable = self.getReward()
			# use the best variation
			if rEnable >= rDisable:
				host.enable = True
			else:
				host.enable = False