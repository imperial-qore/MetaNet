from .Provisioner import *
from provisioner.src.utils import *
from provisioner.src.opt import *

class ACOARIMAProvisioner(Provisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'ARIMA'
		self.search = ACO
		self.model_loaded = False
		self.window_buffer = []
		self.window = None

	def load_model(self):
		self.feats = len(self.env.hostlist)
		self.model_loaded = True

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		predips = self.host_util
		opt = self.search(predips, self.env)
		decisions = opt.search()
		for i, decision in enumerate(decisions):
			self.env.hostlist[i].enable = True if decision == 1 else False