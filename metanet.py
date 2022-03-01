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
	opts.type, opts.env = 2, 'Azure'

	# Generate trace from schedulers
	schedulers = ['GOSH', 'GOBI', 'GA', 'GRAF', 'DecisionNN', 'ACOLSTM']
	for sched in schedulers:
		if os.path.exists('data/'+sched): continue
		generateTrace(sched, 10)

	# Train MetaNet
	model = trainModel(HOSTS, schedulers)
	exit()


	# Run trained scheduler using tuned scheduler
	opts.env, opts.model = 'Azure', 'GOBI'
	datacenter, workload, scheduler, env, stats = initalizeEnvironment(opts.env, int(opts.type), opts.model)
	scheduler.model = tunedModel

	for step in range(NUM_SIM_STEPS):
		print(color.GREEN+"Execution Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)

	datacenter.cleanup()
	saveStats(stats, dirname)




