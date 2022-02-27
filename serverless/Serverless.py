from .node.Node import *
from .task.Task import *
from .datacenter.server.controller import *
from utils.Utils import *
from time import time, sleep
import multiprocessing, os
from joblib import Parallel, delayed
from copy import deepcopy
import numpy as np

num_cores = multiprocessing.cpu_count()

class Serverless():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, Scheduler, Workload, IntervalTime, hostinit, env):
		self.hostlimit = len(hostinit)
		self.scheduler = Scheduler
		self.scheduler.setEnvironment(self)
		self.workload = Workload
		self.workload.setEnvironment(self)
		self.hostlist = []
		self.completedtasklist = []
		self.activetasklist = []
		self.waitinglist = []
		self.intervaltime = IntervalTime
		self.interval = 0
		self.inactiveContainers = []
		self.stats = None
		self.environment = env
		self.addHostlistInit(hostinit)
		self.globalStartTime = time()
	
	def addHostInit(self, IP, IPS, RAM, Disk, Bw, Powermodel):
		assert len(self.hostlist) < self.hostlimit
		host = Node(len(self.hostlist),IP,IPS, RAM, Disk, Bw, Powermodel, self)
		self.hostlist.append(host)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IP, IPS, RAM, Disk, Bw, Powermodel in hostList:
			self.addHostInit(IP, IPS, RAM, Disk, Bw, Powermodel)

	def getTasksOfHost(self, hostID):
		tasks = []
		for task in self.activetasklist:
			if task.hostid == hostID:
				tasks.append(task.id)
		return tasks

	def getTaskByCID(self, creationID):
		return self.functions[creationID]

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	def getNumActiveTasks(self):
		return len(self.activetasklist)

	def getTasksofHost(self, hostID):
		return [t.creationID for t in self.activetasklist if t.hostid == hostID]

	def getTasksInHosts(self):
		return [len(self.getTasksofHost(host)) for host in range(self.hostlimit)]

	def visualSleep(self, t):
		total = str(t%60)+" sec"
		for i in range(int(t)):
			print("\r>> Interval timer "+str(i%60)+" sec of "+total, end=' ')
			sleep(1)
		sleep(t % 1)
		print()

	def canRun(self, task):
		if len(task.precedence) == 0: return True
		done = [False] * len(task.precedence)
		for t in self.completedtasklist:
			if t.creationID == task.creationID and t.taskID in task.precedence:
				done[t.taskID] = True
		return np.all(done)

	def destroyCompletedTasks(self):
		toremove = []
		for i, task in enumerate(self.activetasklist):
			outputexist = [os.path.exists(path) for path in task.output_imgs]
			if np.all(outputexist):
				task.destroy()
				toremove.append(task)
				self.completedtasklist.append(task)
				if task.choice in ['semantic', 'compression']:
					delfiles(task.creationID, task.taskID)
				# if task.choice == 'layer' and task.taskID == 1:
				# 	delfiles(task.creationID)
		for task in toremove: self.activetasklist.remove(task)
		return toremove

	def parallelizedFunc(self, i):
		self.hostlist[i].updateUtilizationMetrics()

	def allocateInit(self, newtasklist, decision):
		self.interval += 1
		start = time(); deployed = 0
		for i, hid in enumerate(decision):
			task = newtasklist[i]
			if self.canRun(task) and self.hostlist[hid].enable:
				task.hostid = hid; task.startAt = self.interval; task.runTask(self.getHostByID(hid).ip)	
				self.activetasklist.append(task); deployed += 1
			else:
				self.waitinglist.append(task)	
		self.visualSleep(self.intervaltime)
		Parallel(n_jobs=num_cores, backend='threading')(delayed(self.parallelizedFunc)(i) for i in list(range(len(self.hostlist))))
		return deployed

	def simulationStep(self, newtasklist, decision):
		self.interval += 1
		start = time(); deployed = 0; toremove = []
		decisionwaiting, decisionnew = decision[:len(self.waitinglist)], decision[len(self.waitinglist):]
		for i, hid in enumerate(decisionwaiting):
			task = self.waitinglist[i]
			if self.canRun(task) and self.hostlist[hid].enable:
				task.hostid = hid; task.startAt = self.interval; task.runTask(self.getHostByID(hid).ip)	
				toremove.append(task); self.activetasklist.append(task); deployed += 1
		for i, hid in enumerate(decisionnew):
			task = newtasklist[i]
			if self.canRun(task) and self.hostlist[hid].enable:
				task.hostid = hid; task.startAt = self.interval; task.runTask(self.getHostByID(hid).ip)	
				self.activetasklist.append(task); deployed += 1
			else:
				self.waitinglist.append(task)	
		for task in toremove: self.waitinglist.remove(task)
		self.visualSleep(self.intervaltime)
		Parallel(n_jobs=num_cores, backend='threading')(delayed(self.parallelizedFunc)(i) for i in list(range(len(self.hostlist))))
		return deployed
