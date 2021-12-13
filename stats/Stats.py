import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plt.style.use(['science'])
plt.rcParams["text.usetex"] = False

class Stats():
	def __init__(self, Environment, WorkloadModel, Datacenter, Scheduler):
		self.env = Environment
		self.env.stats = self
		self.workload = WorkloadModel
		self.datacenter = Datacenter
		self.scheduler = Scheduler
		self.initStats()

	def initStats(self):	
		self.hostinfo = []
		self.workloadinfo = []
		self.alltaskinfo = []
		self.metrics = []
		self.schedulerinfo = []
		self.deciderinfo = []

	def saveHostInfo(self):
		hostinfo = dict()
		hostinfo['interval'] = self.env.interval
		hostinfo['cpu'] = [host.getCPU() for host in self.env.hostlist]
		hostinfo['enable'] = [int(host.enable) for host in self.env.hostlist]
		hostinfo['numcontainers'] = [len(self.env.getTasksofHost(i)) for i,host in enumerate(self.env.hostlist)]
		hostinfo['power'] = [host.getPower() for host in self.env.hostlist]
		hostinfo['baseips'] = [host.getBaseIPS() for host in self.env.hostlist]
		hostinfo['ipsavailable'] = [host.getIPSAvailable() for host in self.env.hostlist]
		hostinfo['ipscap'] = [host.ipsCap for host in self.env.hostlist]
		hostinfo['apparentips'] = [host.getApparentIPS() for host in self.env.hostlist]
		hostinfo['ram'] = [host.getCurrentRAM() for host in self.env.hostlist]
		hostinfo['ramavailable'] = [host.getRAMAvailable() for host in self.env.hostlist]
		hostinfo['disk'] = [host.getCurrentDisk() for host in self.env.hostlist]
		hostinfo['diskavailable'] = [host.getDiskAvailable() for host in self.env.hostlist]
		self.hostinfo.append(hostinfo)

	def saveWorkloadInfo(self, numdep):
		workloadinfo = dict()
		workloadinfo['interval'] = self.env.interval
		workloadinfo['completedtasks'] = len(self.env.completedtasklist)
		workloadinfo['activetasks'] = len(self.env.activetasklist)
		workloadinfo['totaltasks'] = len(self.env.completedtasklist) + len(self.env.activetasklist)
		if self.workloadinfo != []:
			workloadinfo['newtasks'] = workloadinfo['totaltasks'] - self.workloadinfo[-1]['totaltasks'] 
		else:
			workloadinfo['newtasks'] = workloadinfo['totaltasks']
		workloadinfo['deployed'] = numdep
		workloadinfo['inqueue'] = len(self.env.waitinglist)
		self.workloadinfo.append(workloadinfo)

	def saveAllTaskInfo(self):
		self.alltaskinfo = []
		for task in self.env.completedtasklist:
			taskinfo = dict()
			taskinfo['createAt'] = task.createAt
			taskinfo['startAt'] = task.startAt
			taskinfo['destroyAt'] = task.destroyAt
			taskinfo['creationID'] = task.creationID
			taskinfo['taskID'] = task.taskID
			taskinfo['application'] = task.application
			taskinfo['hostid'] = task.hostid
			taskinfo['choice'] = task.choice
			taskinfo['sla'] = task.sla
			self.alltaskinfo.append(taskinfo)

	def saveMetrics(self, destroyed):
		metrics = dict()
		metrics['interval'] = self.env.interval
		metrics['numdestroyed'] = len(destroyed)
		metrics['energy'] = [host.getPower()*self.env.intervaltime for host in self.env.hostlist]
		metrics['energytotalinterval'] = np.sum(metrics['energy'])
		metrics['energypertaskinterval'] = np.sum(metrics['energy'])/self.env.getNumActiveTasks()
		metrics['responsetime'] = [c.destroyAt - c.createAt for c in destroyed]
		metrics['avgresponsetime'] = np.average(metrics['responsetime']) if len(destroyed) > 0 else 0
		metrics['exectime'] = [c.destroyAt - c.startAt for c in destroyed]
		metrics['avgexectime'] = np.average(metrics['exectime']) if len(destroyed) > 0 else 0
		metrics['waittime'] = [c.startAt - c.createAt for c in destroyed]
		metrics['avgwaittime'] = np.average(metrics['waittime']) if len(destroyed) > 0 else 0
		metrics['slaviolations'] = len(np.where([c.destroyAt > c.sla + c.createAt for c in destroyed])[0])
		metrics['slaviolationspercentage'] = metrics['slaviolations'] * 100.0 / len(destroyed) if len(destroyed) > 0 else 0
		self.metrics.append(metrics)

	def saveSchedulerInfo(self, decision, schedulingtime):
		schedulerinfo = dict()
		schedulerinfo['interval'] = self.env.interval
		schedulerinfo['decision'] = [int(d) for d in decision]
		schedulerinfo['schedule'] = [(c.creationID, c.taskID, c.getHostID()) for c in self.env.activetasklist]
		schedulerinfo['schedulingtime'] = schedulingtime
		self.schedulerinfo.append(schedulerinfo)

	def saveDeciderInfo(self, newtasklist):
		deciderinfo = dict()
		deciderinfo['interval'] = self.env.interval
		deciderinfo['applications'] = [task.application for task in newtasklist]
		deciderinfo['choices'] = [task.choice for task in newtasklist]
		self.deciderinfo.append(deciderinfo)

	def saveStats(self, numdep, destroyed, newtasklist, decision, schedulingtime):	
		self.saveHostInfo()
		self.saveWorkloadInfo(numdep)
		self.saveAllTaskInfo()
		self.saveMetrics(destroyed)
		self.saveSchedulerInfo(decision[-len(newtasklist):] if newtasklist else [], schedulingtime)
		self.saveDeciderInfo(newtasklist)

	########################################################################################################

	def generateGraphsWithInterval(self, dirname, listinfo, obj, metric, metric2=None):
		fig, axes = plt.subplots(len(listinfo[0][metric]), 1, sharex=True,figsize=(4, 0.5*len(listinfo[0][metric])))
		axes = [axes] if len(listinfo[0][metric]) == 1 else axes
		title = obj + '_' + metric + '_with_interval' 
		totalIntervals = len(listinfo)
		x = list(range(totalIntervals))
		metric_with_interval = []; metric2_with_interval = []
		ylimit = 0; ylimit2 = 0
		for hostID in range(len(listinfo[0][metric])):
			metric_with_interval.append([listinfo[interval][metric][hostID] for interval in range(totalIntervals)])
			ylimit = max(ylimit, max(metric_with_interval[-1]))
			if metric2:
				metric2_with_interval.append([listinfo[interval][metric2][hostID] for interval in range(totalIntervals)])
				ylimit2 = max(ylimit2, max(metric2_with_interval[-1]))
		for hostID in range(len(listinfo[0][metric])):
			axes[hostID].set_ylim(0, max(ylimit, ylimit2))
			axes[hostID].plot(x, metric_with_interval[hostID])
			if metric2:
				axes[hostID].plot(x, metric2_with_interval[hostID])
			axes[hostID].set_ylabel(obj[0].capitalize()+" "+str(hostID))
			axes[hostID].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + title + '.pdf')

	def generateMetricsWithInterval(self, dirname):
		fig, axes = plt.subplots(8, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.metrics)))
		res = {}
		for i,metric in enumerate(['numdestroyed', 'energytotalinterval', 'energypertaskinterval', 'avgresponsetime',\
			 'avgexectime', 'avgwaittime', 'slaviolations', 'slaviolationspercentage']):
			metric_with_interval = [self.metrics[i][metric] for i in range(len(self.metrics))] if metric != 'waittime' else \
				[sum(self.metrics[i][metric]) for i in range(len(self.metrics))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric, fontsize=5)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
			res[metric] = sum(metric_with_interval)
			print("Summation ", metric, " = ", res[metric])
		print('Average energy (sum energy interval / sum numdestroyed) = ', res['energytotalinterval']/res['numdestroyed'])
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Metrics' + '.pdf')

	def generateWorkloadWithInterval(self, dirname):
		fig, axes = plt.subplots(6, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.workloadinfo)))
		for i,metric in enumerate(['completedtasks', 'activetasks', 'totaltasks', 'newtasks', 'deployed', 'inqueue']):
			metric_with_interval = [self.workloadinfo[i][metric] for i in range(len(self.workloadinfo))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Workload' + '.pdf')
	
	def generateScheduleWithInterval(self, dirname):
		fig, axes = plt.subplots(1, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.schedulerinfo)))
		for i,metric in enumerate(['schedulingtime']):
			metric_with_interval = [self.schedulerinfo[i][metric] for i in range(len(self.schedulerinfo))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Workload' + '.pdf')

	########################################################################################################

	def generateCompleteDataset(self, dirname, data, name):
		title = name + '_dataset' 
		metric_with_interval = []
		headers = list(data[0].keys())
		for datum in data:
			metric_with_interval.append([datum[value] for value in datum.keys()])
		df = pd.DataFrame(metric_with_interval, columns=headers)
		df.to_csv(dirname + '/' + title + '.csv', index=False)

	def generateSimpleMetricsDatasetWithInterval(self, dirname, metric):
		title = metric + '_' + 'with_interval' 
		totalIntervals = len(self.hostinfo)
		metric_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append(np.mean(self.metrics[interval][metric]))
		df = pd.DataFrame(metric_with_interval)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateSimpleHostDatasetWithInterval(self, dirname, metric):
		title = metric + '_' + 'with_interval' 
		totalIntervals = len(self.hostinfo)
		metric_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append([self.hostinfo[interval][metric][hostID] for hostID in range(len(self.hostinfo[0][metric]))])
		df = pd.DataFrame(metric_with_interval)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateSimpleDeciderDatasetWithInterval(self, dirname):
		title = 'decider' + '_' + 'with_interval' 
		totalIntervals = len(self.deciderinfo)
		metric_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append(self.deciderinfo[interval]['applications'] + self.deciderinfo[interval]['choices'])
		df = pd.DataFrame(metric_with_interval)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)
	
	def generateSimpleSchedulerDatasetWithInterval(self, dirname):
		title = 'scheduler' + '_' + 'with_interval' 
		totalIntervals = len(self.deciderinfo)
		metric_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append(self.deciderinfo[interval]['applications'] + self.schedulerinfo[interval]['decision'])
		df = pd.DataFrame(metric_with_interval)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateGraphs(self, dirname):
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'cpu')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'numcontainers')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'power')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'enable')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'baseips', 'apparentips')
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'ipscap', 'apparentips')
		self.generateMetricsWithInterval(dirname)
		self.generateWorkloadWithInterval(dirname)

	def generateDatasets(self, dirname):
		self.generateSimpleHostDatasetWithInterval(dirname, 'cpu')
		self.generateSimpleHostDatasetWithInterval(dirname, 'enable')
		self.generateSimpleMetricsDatasetWithInterval(dirname, 'energy')
		self.generateSimpleDeciderDatasetWithInterval(dirname)
		self.generateSimpleSchedulerDatasetWithInterval(dirname)
		
	def generateCompleteDatasets(self, dirname):
		self.generateCompleteDataset(dirname, self.hostinfo, 'hostinfo')
		self.generateCompleteDataset(dirname, self.workloadinfo, 'workloadinfo')
		self.generateCompleteDataset(dirname, self.metrics, 'metrics')
		self.generateCompleteDataset(dirname, self.alltaskinfo, 'alltaskinfo')
		self.generateCompleteDataset(dirname, self.schedulerinfo, 'schedulerinfo')
		self.generateCompleteDataset(dirname, self.deciderinfo, 'deciderinfo')
	