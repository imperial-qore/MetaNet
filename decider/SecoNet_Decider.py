from .Decider import *
from provisioner.src.utils import *

class SecoNetDecider(Decider):
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

	def decision(self, workflowlist):
		if not self.model_loaded: self.load_model()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		window = self.host_util.view(1, 1, self.feats)
		_, pred = self.model.predwindow(window, window)
		window_next = pred.view(1, 1, self.feats)
		memory, _ = self.model.predwindow(window_next, window_next)
		window_next = pred.view(1, 1, self.feats)
		results = []
		for CreationID, interval, SLA, application in workflowlist:
			inp = one_hot(application, self.fn_names)
			choice = self.choices[torch.argmax(self.model.forward_decider(memory, inp)).item()]
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results