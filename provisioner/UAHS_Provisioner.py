from .Provisioner import *
from provisioner.src.utils import *
from provisioner.src.opt import *
from provisioner.src.hgp import *

class UAHSProvisioner(Provisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'UAHS'
		self.search = StochasticSearch
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.model = None

	def load_model(self):
		# Load model
		self.feats = len(self.env.hostlist)
		# Load dataset
		dataset, _ = load_cpu_dataset(self.feats)
		# Train model 
		X = np.array([np.array(i).reshape(-1) for i in dataset])
		y = np.roll(X, 1, axis=0)
		kernel_hetero = C(1.0, (1e-10, 1000)) * RBF(0.5, (0.00, 100.0)) 
		self.model = [GaussianProcessRegressor(kernel=kernel_hetero, alpha=0.01) for _ in range(X.shape[1])]
		file_path = base_url + f'checkpoints/UAHS.ckpt'
		if os.path.exists(file_path):
			print(color.GREEN+"Loading pre-trained model: UAHS"+color.ENDC)
			with open(file_path, 'rb') as f:
				self.model = pickle.load(f)
		else:
			print(color.GREEN+"Creating new model: UAHS"+color.ENDC)
			for i in range(X.shape[1]):
				dx, dy = X[:, i].reshape(X.shape[0], -1), y[:, i].reshape(X.shape[0], -1)
				self.model[i] = self.model[i].fit(dx, dy)
			with open(file_path, 'wb') as f:
				pickle.dump(self.model, f)
		self.model_loaded = True

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = np.array([h.getCPU() for h in self.env.hostlist]) / 100
		window = self.host_util; predips, stdips = [], []
		for i in range(window.shape[0]):
			dx = np.array([window[i]])
			p, s = self.model[i].predict(dx.reshape(1, -1), return_std=True)
			predips.append(p[0][0]); stdips.append(s[0])
		opt = self.search(predips, stdips, self.env)
		decisions = opt.search()
		for i, decision in enumerate(decisions):
			self.env.hostlist[i].enable = True if decision == 1 else False