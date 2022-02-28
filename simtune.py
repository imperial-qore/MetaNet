from main import *
import random

def perturb_params(params):
	for i in range(len(params)):
		for j in range(len(params[i])):
			params[i][j] = max(0, params[i][j] * (1 + (random.random() - 0.5) / 10))
	return params	

def runModel(model, steps = NUM_SIM_STEPS, dirname = 'real'):
	global opts; opts.model = model
	# Initialize Environment
	datacenter, workload, scheduler, env, stats = initalizeEnvironment(opts.env, int(opts.type), opts.model)

	# Execute steps
	for step in range(steps):
		print(color.GREEN+"Execution Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)
		if 'surrogate' in dirname and step % 10 == 0:
			params = env.generateParams()
			params = perturb_params(params)
			env.updateParams(params)

	# Cleanup and save results
	if env.__class__.__name__ == 'Serverless': datacenter.cleanup()
	return stats

def generateRandomTrace(env, dirname):
	# return saved stats if this is a prerun host set
	global opts
	opts.env = env
	os.makedirs(f"data/{dirname}", exist_ok=True)
	stats = runModel('Random', NUM_SIM_STEPS if env == 'Azure' else NUM_SIM_STEPS * 2, dirname)
	stats.generateSimpleHostDatasetWithInterval(f'data/{dirname}/', 'cpu')
	stats.generateSimpleMetricsDatasetWithInterval(f'data/{dirname}/', 'avgresponsetime')
	stats.generateSimpleMetricsDatasetWithInterval(f'data/{dirname}/', 'energytotalinterval')
	if 'surrogate' in dirname:
		stats.generateSimulatorParamsWithInterval(f'data/{dirname}/')
	return stats

if __name__ == '__main__':
	global opts
	if not os.path.exists("logs"): os.mkdir("logs")
	opts.type, opts.env = 2, 'Azure'

	# Generate trace from Random scheduler
	# realTrace = generateRandomTrace('Azure', 'real')

	# Train a surrogate model with generated trace
	simTrace = generateRandomTrace('Sim', 'sim_surrogate')
	# surrogate = trainModel()
	
	# Update simulator parameters

	# Generate more data trace using simulator

	# Train scheduler using more vertical data

	# Run trained scheduler using scheduler's horizontal data




