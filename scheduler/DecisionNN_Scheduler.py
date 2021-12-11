from .Scheduler import *
from provisioner.src.utils import *
from .src.opt import *

class DecisionNNScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'Attention'
		self.search = LocalSearch
		self.model_loaded = False

	def load_model(self):
		# Assume pretrained model loaded by provisioner
		self.feats = len(self.env.hostlist)
		self.model = self.env.provisioner.model
		self.host_util = self.env.provisioner.host_util
		self.model_loaded = True

	def placement(self, tasks):
		if not self.model_loaded: self.load_model()
		start = time()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		predips = self.model(self.host_util)
		opt = self.search(predips, self.env, tasks)
		decisions = opt.search()
		return decisions, time() - start