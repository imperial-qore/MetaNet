from .Provisioner import *
from provisioner.src.utils import *
from provisioner.src.opt import *

class ACOLSTMProvisioner(Provisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'LSTM'
		self.search = ACO
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.model = None

	def load_model(self):
		# Load model
		self.feats = len(self.env.hostlist)
		self.model, self.optimizer, self.scheduler, epoch, self.loss_list = load_model(self.model_name, self.feats)
		# Load dataset
		train_cpu, test_cpu = load_cpu_dataset(self.feats)
		train_provisioner, test_provisioner = load_provisioner_dataset(self.feats)
		train_decider, test_decider = load_decider_dataset(self.feats)
		train_scheduler, test_scheduler = load_scheduler_dataset(self.feats)
		# Train model 
		if epoch == -1:
			for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
				lossT, lr = backprop(e, self.model, self.optimizer, self.scheduler, train_cpu, train_provisioner, train_decider, train_scheduler)
				lossTest, _ = backprop(e, self.model, self.optimizer, self.scheduler, test_cpu, test_provisioner, test_decider, test_scheduler, False)
				self.loss_list.append((lossT, lossTest, lr))
				tqdm.write(f'Epoch {e},\tTrain Loss = {lossT},\tTest loss = {lossTest}')
				plot_accuracies(self.loss_list, base_url, self.model)
			save_model(self.model, self.optimizer, self.scheduler, e, self.loss_list)
		# Freeze encoder
		freeze(self.model)
		self.model_loaded = True

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		predips = self.model(self.host_util)
		opt = self.search(predips, self.env)
		decisions = opt.search()
		for i, decision in enumerate(decisions):
			self.env.hostlist[i].enable = True if decision == 1 else False