from .Scheduler import *
from .src.provisioner_utils import *
from .src.opt import *

class ACOARIMAScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'ARIMA'
		self.search = ACO
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.model = None

	def load_model(self):
		self.feats = len(self.env.hostlist)
		self.model_loaded = True

	def placement(self, tasks):
		if not self.model_loaded: self.load_model()
		start = time()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		predips = self.host_util
		opt = self.search(predips, self.env)
		decisions = opt.search()
		return decisions, time() - start