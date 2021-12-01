import os, sys, stat
import sys
import optparse
import logging as logger
import configparser
import pickle
import platform
from time import time
from subprocess import call
from os import system, rename

# Framework imports
from serverless.Serverless import *
from serverless.datacenter.AzureDatacenter import *
from serverless.workload.AIBenchWorkload import *

# Provisioner imports
from provisioner.Provisioner import Provisioner
from provisioner.Random_Provisioner import RandomProvisioner

# Decider imports
from decider.Random import RandomDecider
from decider.Layer_Only import LayerOnlyDecider
from decider.Semantic_Only import SemanticOnlyDecider
from decider.Compression_Only import CompressionOnlyDecider

# Scheduler imports
from scheduler.Random import RandomScheduler

# Auxiliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp

usage = "usage: python main.py -e <environment> -m <mode> # empty environment run simulator"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-e", "--environment", action="store", dest="env", default="", 
					help="Environment is Azure or VLAN.")
parser.add_option("-m", "--mode", action="store", dest="mode", default="0", 
					help="Mode is 0 (Create and destroy), 1 (Create), 2 (No op), 3 (Destroy)")
opts, args = parser.parse_args()

# Global constants
NUM_SIM_STEPS = 100
HOSTS = 10
INTERVAL_TIME = 5 # seconds
NEW_CONTAINERS = 0
HOSTS_IP = []

def initalizeEnvironment(environment, mode):
	# Initialize simple fog datacenter
	''' Can be AzureDatacenter '''
	datacenter = eval(environment+'Datacenter(mode)')
	hostlist = datacenter.generateHosts()
	
	# Initialize workload
	''' Can be AIBench '''
	workload = AIBenchWorkload(NEW_CONTAINERS, 1.5)

	# Initialize decider
	''' Can be Random '''
	decider = RandomDecider() 

	# Initialize scheduler
	''' Can be Random '''
	scheduler = RandomScheduler() 

	# Initialize Environment
	env = Serverless(scheduler, decider, INTERVAL_TIME, hostlist, environment)

	# Execute first step
	workloadlist = workload.generateNewContainers(env.interval) # New containers info
	newtasklist = decider.decision(workloadlist)
	decision, schedulingTime = scheduler.placement(newtasklist) # Decide placement using task objects
	numdep = env.allocateInit(newtasklist, decision) # Schedule functions
	print("New Tasks Size:", len(newtasklist))
	print("Waiting List Size:", len(env.waitinglist))
	print("Tasks in hosts:", env.getTasksInHosts())
	print("Deployed:", numdep, "of", len(env.waitinglist + newtasklist))
	print("Decision:", decision)

	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)
	# stats.saveStats(deployed, migrations, [], deployed, decision, schedulingTime)
	return datacenter, workload, scheduler, decider, env, stats

def stepSimulation(workload, scheduler, decider, env, stats):
	workloadlist = workload.generateNewContainers(env.interval) # New containers info
	newtasklist = decider.decision(workloadlist)
	decision, schedulingTime = scheduler.placement(env.waitinglist + newtasklist) # Decide placement using task objects
	numdes = env.destroyCompletedTasks()
	numdep = env.simulationStep(newtasklist, decision) # Schedule containers
	print("New Tasks Size:", len(newtasklist))
	print("Waiting List Size:", len(env.waitinglist))
	print("Tasks in hosts:", env.getTasksInHosts())
	print("Deployed:", numdep, "of", len(env.waitinglist + newtasklist))
	print("Destroyed:", numdes, "of", len(env.activetasklist))
	print("Decision:", decision)

	# stats.saveStats(deployed, migrations, destroyed, selected, decision, schedulingTime)

def saveStats(stats, datacenter, workload, env, end=True):
	dirname = "logs/" + datacenter.__class__.__name__
	dirname += "_" + workload.__class__.__name__
	dirname += "_" + str(NUM_SIM_STEPS) 
	dirname += "_" + str(HOSTS)
	dirname += "_" + str(INTERVAL_TIME)
	dirname += "_" + str(NEW_CONTAINERS)
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	stats.generateDatasets(dirname)
	if not end: return
	stats.generateGraphs(dirname)
	stats.generateCompleteDatasets(dirname)
	stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':
	datacenter, workload, scheduler, decider, env, stats = initalizeEnvironment(opts.env, int(opts.mode))

	for step in range(NUM_SIM_STEPS):
		print(color.GREEN+"Execution Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, decider, env, stats)
		# if step % 10 == 0: saveStats(stats, datacenter, workload, env, end = False)

	datacenter.cleanup()
	# saveStats(stats, datacenter, workload, env)

