from .Provisioner import *
from provisioner.src.utils import *
from provisioner.src.opt import *
from pykalman import KalmanFilter

class CESProvisioner(Provisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'CES'
		self.search = LocalSearch
		self.model_loaded = False
		self.window_buffer = []
		self.window_size = 10
		self.predips = None

	def load_model(self):
		# Load model
		self.feats = len(self.env.hostlist)
		self.model = KalmanFilter(n_dim_obs=self.feats, n_dim_state=self.feats)
		self.model_loaded = True

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = np.array([h.getCPU() for h in self.env.hostlist]) / 100
		self.window_buffer.append(self.host_util)
		self.window_buffer = self.window_buffer[-self.window_size:] if len(self.window_buffer) > 1 else self.window_buffer * 2
		kf = self.model.em(self.window_buffer, n_iter=100).smooth(self.window_buffer)[0]
		self.predips = kf[-1]
		opt = self.search(self.predips, self.env)
		decisions = opt.search()
		for i, decision in enumerate(decisions):
			self.env.hostlist[i].enable = True if decision == 1 else False