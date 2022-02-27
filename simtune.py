from main import *

def runModel(model):
	global opts
	opts.model = model

	# Create log directory
	dirname = "logs/" + opts.model
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)

	# Initialize Environment
	datacenter, workload, scheduler, env, stats = initalizeEnvironment(opts.env, int(opts.type), opts.model)

	# Execute steps
	for step in range(NUM_SIM_STEPS):
		print(color.GREEN+"Execution Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)
		if step % 10 == 0: saveStats(stats, dirname, end = False)

	# Cleanup and save results
	if env.__class__.__name__ == 'Serverless':
		datacenter.cleanup()
	saveStats(stats, dirname)
	return stats

def generateRealTrace():
	# return saved stats if this is a prerun host set
	stats = runModel('Random')
	stats.generateSimpleHostDatasetWithInterval('data', 'cpu')
	stats.generateSimpleMetricsDatasetWithInterval('data', 'avgresponsetime')
	stats.generateSimpleMetricsDatasetWithInterval('data', 'energytotalinterval')
	return stats

if __name__ == '__main__':
	global opts
	if not os.path.exists("logs"): os.mkdir("logs")
	opts.type, opts.env = 2, 'Azure'

	# Generate trace from Random scheduler
	realTrace = generateRealTrace()

	# Train a surrogate model
	
	# Update simulator parameters

	# Generate more data trace using simulator

	# Train scheduler using more vertical data

	# Run trained scheduler using scheduler's horizontal data




