from .Scheduler import *
from provisioner.src.utils import *

class SecoNetScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'SecoNet'
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.fn_names = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]

	def load_model(self):
		# Assume pretrained model loaded by provisioner
		self.feats = len(self.env.hostlist)
		self.model = self.env.provisioner.model
		self.model_loaded = True

	def placement(self, tasks):
		if not self.model_loaded: self.load_model()
		start = time()
		memory = self.env.provisioner.memory
		decision = []
		for task in tasks:
			inp = one_hot(task.application, self.fn_names)
			scores = self.model.forward_scheduler(memory, inp).tolist()
			# mask disabled hosts
			for hostID, host in enumerate(self.env.hostlist):
				if not host.enable: scores[hostID] = 0
			decision.append(np.argmax(scores))
		return decision, time() - start