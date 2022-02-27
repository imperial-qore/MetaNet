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

# Scheduler imports
from scheduler.Random import RandomScheduler
from scheduler.CoSim_Scheduler import CoSimScheduler
from scheduler.SciNet_Scheduler import SciNetScheduler
from scheduler.ACOARIMA_Scheduler import ACOARIMAScheduler
from scheduler.ACOLSTM_Scheduler import ACOLSTMScheduler
from scheduler.DecisionNN_Scheduler import DecisionNNScheduler
from scheduler.SemiDirect_Scheduler import SemiDirectScheduler
from scheduler.GRAF_Scheduler import GRAFScheduler
from scheduler.GOBI_Scheduler import GOBIScheduler
from scheduler.GOSH_Scheduler import GOSHScheduler
from scheduler.HASCO_Scheduler import HASCOScheduler
from scheduler.RecSim_Scheduler import RecSimScheduler
from scheduler.CES_Scheduler import CESScheduler

# Auxiliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp

usage = "usage: python main.py -e <environment> -t <type> -m <model>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-e", "--environment", action="store", dest="env", default="Azure", 
					choices=['Azure', 'VLAN'],
					help="Environment is Azure or VLAN.")
parser.add_option("-t", "--type", action="store", dest="type", default="2", 
					choices=['0', '1', '2', '3'],
					help="Type is 0 (Create and destroy), 1 (Create), 2 (No op), 3 (Destroy)")
parser.add_option("-m", "--model", action="store", dest="model", default="Random", 
					choices=['Random', 'CoSim', 'ACOARIMA', 'ACOLSTM', 'DecisionNN', 'SemiDirect',\
						'GRAF', 'UAHS', 'CAHS', 'Narya', 'HASCO', 'RecSim', 'CES', 'SciNet'])
opts, args = parser.parse_args()

# Global constants
NUM_SIM_STEPS = 200
HOSTS = 10
INTERVAL_TIME = 5 # seconds
NEW_TASKS = 0

def initalizeEnvironment(environment, type, prov, dec, sched):
	# Initialize simple fog datacenter
	''' Can be AzureDatacenter '''
	datacenter = eval(environment+'Datacenter(type)')
	hostlist = datacenter.generateHosts()
	
	# Initialize workload
	''' Can be AIBench '''
	workload = AIBenchWorkload(NEW_TASKS, 1.5)

	# Initialize scheduler
	scheduler = eval(sched+'Scheduler()')

	# Initialize Environment
	env = Serverless(scheduler, workload, INTERVAL_TIME, hostlist, environment)

	# Execute first step
	newtasklist = workload.generateNewContainers(env.interval) # New containers info
	decision, schedulingTime = scheduler.placement(newtasklist) # Decide placement using task objects
	numdep = env.allocateInit(newtasklist, decision) # Schedule functions
	print("New Tasks Size:", len(newtasklist))
	print("Waiting List Size:", len(env.waitinglist))
	print("Tasks in hosts:", env.getTasksInHosts())
	print("Deployed:", numdep, "of", len(env.waitinglist + newtasklist))
	printProvisioned(env.hostlist)
	printDecisions(newtasklist)
	print("Schedule:", decision)

	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)
	stats.saveStats(numdep, [], newtasklist, decision, schedulingTime)
	return datacenter, workload, scheduler, env, stats

def stepSimulation(workload, scheduler, env, stats):
	newtasklist = workload.generateNewContainers(env.interval) # New containers info
	decision, schedulingTime = scheduler.placement(env.waitinglist + newtasklist) # Decide placement using task objects
	destroyed = env.destroyCompletedTasks()
	numdep = env.simulationStep(newtasklist, decision) # Schedule containers
	print("New Tasks Size:", len(newtasklist))
	print("Waiting List Size:", len(env.waitinglist))
	print("Tasks in hosts:", env.getTasksInHosts())
	print("Deployed:", numdep, "of", len(env.waitinglist + newtasklist))
	print("Destroyed:", len(destroyed), "of", len(env.activetasklist))
	printProvisioned(env.hostlist)
	printDecisions(newtasklist)
	print("Schedule:", decision)

	stats.saveStats(numdep, destroyed, newtasklist, decision, schedulingTime)

def saveStats(stats, dirname, end=True):
	stats.generateDatasets(dirname)
	if not end: return
	stats.generateGraphs(dirname); stats.generateCompleteDatasets(dirname)
	stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
	with open(dirname + '/stats.pk', 'wb') as handle:  pickle.dump(stats, handle)

if __name__ == '__main__':
	# Initialize Environment
	datacenter, workload, scheduler, env, stats = initalizeEnvironment(opts.env, int(opts.type), *decompose(opts.model))

	# Create log directory
	dirname = "logs/" + opts.model
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)

	# Execute steps
	for step in range(NUM_SIM_STEPS):
		print(color.GREEN+"Execution Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)
		if step % 10 == 0: saveStats(stats, dirname, end = False)

	# Cleanup and save results
	datacenter.cleanup()
	saveStats(stats, dirname)

