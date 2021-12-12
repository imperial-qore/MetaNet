from .Provisioner import *
from provisioner.src.utils import *

class HASCOProvisioner(Provisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'HASCO'
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.memory = None
		self.fn_names = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]

	def load_model(self):
		# Load model
		self.feats = len(self.env.hostlist)
		self.model, self.optimizer, self.scheduler, epoch, self.loss_list = load_model(self.model_name, self.feats)
		# Load dataset
		train_cpu, test_cpu = load_cpu_dataset(self.feats)
		train_provisioner, test_provisioner = load_provisioner_dataset(self.feats)
		train_decider, test_decider = load_decider_dataset(self.feats)
		train_scheduler, test_scheduler = load_scheduler_dataset(self.feats)
		train_energy, test_energy, self.emax = load_energy_dataset(self.feats)
		# Train model 
		if epoch == -1:
			for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
				lossT, lr = backprop(e, self.model, self.optimizer, self.scheduler, train_cpu, train_provisioner, train_decider, train_scheduler, True, train_energy)
				lossTest, _ = backprop(e, self.model, self.optimizer, self.scheduler, test_cpu, test_provisioner, test_decider, test_scheduler, False, test_energy)
				self.loss_list.append((lossT, lossTest, lr))
				tqdm.write(f'Epoch {e},\tTrain Loss = {lossT},\tTest loss = {lossTest}')
				plot_accuracies(self.loss_list, base_url, self.model)
			save_model(self.model, self.optimizer, self.scheduler, e, self.loss_list)
		# Freeze encoder
		freeze(self.model)
		self.model_loaded = True

	def testall(self, cpu, prov=None, app=None, dec=None, sched=None):
		scores = []; 
		p_choices, app_choices, d_choices, s_choices = [0, 1], self.fn_names, ['layer', 'semantic', 'compression'], list(range(self.feats))
		for p_choice in p_choices:
			if not prov is None and p_choice != prov: continue
			p = one_hot(p_choice, p_choices)
			for app_choice in app_choices:
				if not app is None and app_choice != app: continue
				app = one_hot(app_choice, app_choices)
				for d_choice in d_choices:
					if not dec is None and d_choice != dec: continue
					d = one_hot(d_choice, d_choices)
					for s_choice in s_choices:
						if not sched is None and s_choice != sched: continue
						sched = one_hot(s_choice, s_choices)
						scores.append(self.model(cpu, p, app, d, sched))
		return np.mean(scores)

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		cpu = self.host_util
		for i in range(len(self.env.hostlist)):
			scores = [self.testall(cpu, prov=i) for i in [0, 1]]
			todo = np.argmax(scores)
			self.env.hostlist[i].enable = True if todo == 1 else False