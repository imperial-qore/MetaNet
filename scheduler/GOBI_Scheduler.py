from .Scheduler import *
from .src.gobi import *

class GOBIScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'GOBI'
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.alpha, self.beta, self.delta = 0.3, 0.3, 0.3
		self.rt_dict, self.sla_dict = get_dicts()
		self.estimateTaskIPS = get_estimate_ips(); 
		self.fn_names = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]

	def load_model(self):
		# Load model
		self.feats = len(self.env.hostlist)
		self.model, self.optimizer, self.scheduler, epoch, self.loss_list = load_model(self.model_name, self.feats)
		# Load dataset
		train_cpu, test_cpu = load_cpu_dataset(self.feats)
		train_scheduler, test_scheduler = load_scheduler_dataset(self.feats)
		train_energy, test_energy, self.emax = load_energy_dataset(self.feats)
		# Train model 
		if epoch == -1:
			for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
				lossT, lr = backprop(e, self.model, self.optimizer, self.scheduler, train_cpu, train_scheduler, train_energy)
				lossTest, _ = backprop(e, self.model, self.optimizer, self.scheduler, test_cpu, test_scheduler, train_energy, False)
				self.loss_list.append((lossT, lossTest, lr))
				tqdm.write(f'Epoch {e},\tTrain Loss = {lossT},\tTest loss = {lossTest}')
				plot_accuracies(self.loss_list, base_url, self.model)
			save_model(self.model, self.optimizer, self.scheduler, e, self.loss_list)
		# Freeze encoder
		freeze(self.model)
		self.model_loaded = True

	def runSimpleSimulation(self, task, hostID):
		ehosts = []; 
		application, choice = task.application, task.choice
		for host in self.env.hostlist:
			ehosts.append(host.getPowerFromIPS(host.ips + self.estimateTaskIPS) if host.id == hostID else host.getPower())
		e = np.sum(ehosts)
		e = e / self.emax # normalize
		sla = self.sla_dict[application]
		rt = np.random.normal(loc = self.rt_dict[application][choice][0], scale = self.rt_dict[application][choice][1])
		if self.env.hostlist[hostID].getIPSAvailable() > self.estimateTaskIPS:
			rt /= 2
		sla_v = (rt <= sla) + 0
		return 1 - (self.alpha * e + self.delta * sla_v)

	def placement(self, tasks):
		if not self.model_loaded: self.load_model()
		start = time()
		cpu = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		decision = []
		for task in tasks:
			app = torch.FloatTensor(one_hot(task.application, self.fn_names))
			scores = [self.runSimpleSimulation(task, hostID) for hostID, _ in enumerate(self.env.hostlist)]
			for hostID, host in enumerate(self.env.hostlist): # do not schedule on disabled hosts
				if not host.enable: scores[hostID] = -1000
			init_decision = np.argmax(scores)
			sched = torch.tensor(one_hot(float(init_decision)//1, range(self.feats)), dtype=torch.float, requires_grad=True)
			sched = gobi_opt(self.model, cpu, app, sched)
			decision.append(torch.argmax(sched).item())
		return decision, time() - start