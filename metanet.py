from main import *
from surrogate.utils import *

def runModel(model, steps = NUM_SIM_STEPS):
	global opts; opts.model = model
	# Initialize Environment
	datacenter, workload, scheduler, env, stats = initalizeEnvironment(opts.env, int(opts.type), opts.model)

	# Execute steps
	for step in range(steps):
		print(color.GREEN+"Execution Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)

	# Cleanup and save results
	if env.__class__.__name__ == 'Serverless': datacenter.cleanup()
	return stats

def generateTrace(model, steps = NUM_SIM_STEPS):
	# return saved stats if this is a prerun host set
	os.makedirs(f"data/{model}", exist_ok=True)
	stats = runModel(model, steps)
	stats.generateSimpleHostDatasetWithInterval(f'data/{model}/', 'cpu')
	stats.generateSimpleMetricsDatasetWithInterval(f'data/{model}/', 'avgresponsetime')
	stats.generateSchedulerTimeDatasetWithInterval(f'data/{model}/')
	return stats

if __name__ == '__main__':
	global opts
	dirname = "logs/MetaNet"
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	opts.type, opts.env, opts.model = 2, 'Azure', 'GOBI'
	NUM_TASKS = 1000; WORKER_COST_PS = 0.008; BROKER_COST_PS = 0.000016

	# Generate trace from schedulers
	schedulers = ['GOSH', 'GOBI', 'GA', 'GRAF', 'DecisionNN', 'ACOLSTM']
	for sched in schedulers:
		if os.path.exists('data/'+sched): continue
		generateTrace(sched, NUM_SIM_STEPS)

	# Train MetaNet
	model, range_rt, range_st = trainModel(HOSTS, schedulers)

	# Run experiment using MetaNet
	datacenter, workload, scheduler, env, stats = initalizeEnvironment(opts.env, int(opts.type), opts.model)
	scheduler_objs = []
	for sched in schedulers:
		s_obj = eval(sched+'Scheduler()')
		s_obj.setEnvironment(env)
		scheduler_objs.append(s_obj)

	for step in range(NUM_SIM_STEPS):
		print(color.GREEN+"Execution Interval:", step, color.ENDC)
		# Select scheduler
		cpu = torch.tensor([h.getCPU() for h in env.hostlist], dtype=torch.float64) / 100
		predr, preds = model(cpu)
		predr, preds = predr.detach().numpy(), preds.detach().numpy()
		predr, preds = predr  * WORKER_COST_PS, preds * BROKER_COST_PS * NUM_SIM_STEPS * INTERVAL_TIME
		final_costs = predr + preds
		optimum_scheduler_index = np.argmin(final_costs)
		scheduler = scheduler_objs[optimum_scheduler_index]

		# Run simulation step
		stepSimulation(workload, scheduler, env, stats)

	datacenter.cleanup()
	saveStats(stats, dirname)




