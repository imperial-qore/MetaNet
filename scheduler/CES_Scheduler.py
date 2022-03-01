from .Scheduler import *
from .src.provisioner_utils import *
from .src.opt import *

class CESScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'CES'
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
		predips = self.env.provisioner.predips
		opt = self.search(predips, self.env, tasks)
		decisions = opt.search()
		return decisions, time() - start