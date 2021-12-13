from .Provisioner import *
from provisioner.src.utils import *

class SecoNetProvisioner(Provisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'SecoNet'
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.memory = None
		self.allpowermodels = ['PMB2s', 'PMB2ms', 'PMB4ms', 'PMB8ms']
		costs = np.array([0.08, 0.12, 0.17, 0.33]) / 12
		ipscaps = [2019, 2019, 4029, 16111]
		self.costdict = dict(zip(self.allpowermodels, costs))
		self.ipscaps = dict(zip(self.allpowermodels, ipscaps))
		self.gamma = 0.2 # weight of cost w.r.t utilization ratio

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

	def updateMetrics(self):
		self.costs = [self.costdict[host.powermodel.__class__.__name__] if host.enable else 0 for host in self.env.hostlist]
		self.util = [(host.ips / host.ipsCap) if host.enable else 0 for host in self.env.hostlist]

	def getReward(self):
		return (np.sum(self.util) - self.gamma * np.sum(self.costs)) * (0 if np.sum(self.util) == 0 else 1)

	def backup_provision(self):
		for host in self.env.hostlist:
			# run A/B testing
			host.enable = True
			self.updateMetrics()
			rEnable = self.getReward()
			host.enable = False
			self.updateMetrics()
			rDisable = self.getReward()
			# use the best variation
			host.enable = rEnable >= rDisable

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		window = self.host_util.view(1, 1, self.feats)
		_, pred = self.model.predwindow(window, window)
		window_next = pred.view(1, 1, self.feats)
		memory, _ = self.model.predwindow(window_next, window_next)
		self.memory = memory # save for decider and scheduler
		decisions = [self.model.forward_provisioner(memory, i) for i in self.host_util]
		hosts_active = 0
		for i, decision in enumerate(decisions):
			todo = torch.argmax(decision).item()
			hosts_active += todo
			self.env.hostlist[i].enable = True if todo == 1 else False
		if hosts_active < self.feats // 4: self.backup_provision()