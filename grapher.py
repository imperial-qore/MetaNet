import matplotlib.pyplot as plt
import matplotlib
import itertools
import statistics
import pickle
import numpy as np
import pandas as pd
import scipy.stats
from stats.Stats import *
import seaborn as sns
from pprint import pprint
from utils.Utils import *
from utils.ColorUtils import *
import os
import fnmatch
from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
size = (2.9, 2.5)
option = 0
INTERVAL_TIME = 5 # seconds

def fairness(l):
	a = 1 / (np.mean(l)-(scipy.stats.hmean(l)+0.001)) # 1 / slowdown i.e. 1 / (am - hm)
	if a: return a
	return 0

def jains_fairness(l):
	a = np.sum(l)**2 / (len(l) * np.sum(l**2)) # Jain's fairness index
	if a: return a
	return 0

def reduce(l):
	n = 5
	res = []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
	return res

def fstr(val):
	# return "{:.2E}".format(val)
	return "{:.2f}".format(val)

def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    h = scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


SAVE_PATH = 'results/' + '/'
os.makedirs(SAVE_PATH, exist_ok=True)

plt.rcParams["figure.figsize"] = 3.3,2.5

Models = os.listdir('./logs/')
Models = [i for i in Models if '_' not in i]
Models.remove('CES')
sla_baseline = Models[0]
ModelsXticks = Models
rot = 90
Colors = ['red', 'blue', 'green', 'orange', 'orchid', 'pink', 'cyan'] * 2
apps = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]
accs = dict(zip(apps, [0.7, 0.8, 0.85, 0.92, 0.97, 0.89, 0.76]))
choices = ['layer', 'semantic', 'compression']
choice_multiplier = dict(zip(choices, [1, 0.8, 0.6]))

yLabelsStatic = ['Total Energy (Kilowatt-hr)', 'Average Energy (Kilowatt-hr)', 'Interval Energy (Kilowatt-hr)', \
'Average Interval Energy (Kilowatt-hr)', 'Number of completed tasks', 'Cost per container (US Dollars)', \
'Number of completed workflows per application', 'Number of completed tasks per interval', \
'Average Response Time (seconds)', 'Average Execution Time (seconds)', 'Average Waiting Time (seconds)', \
'Amortized Response Time (seconds)', 'Average Response Time (seconds) per application', \
'Average Execution Time (seconds) per application', 'Average Waiting Time (seconds) per application', \
'Amortized Response Time (seconds) per application', "Fairness (Jain's index)", 'Fairness per application', \
'Amortized Workflow Response Time (seconds)', 'Amortized Workflow Response Time per application (seconds)', \
'Amortized Workflow Waiting Time (seconds)', 'Amortized Workflow Waiting Time per application (seconds)', \
'Average Workflow Accuracy', 'Average Workflow Accuracy per application', 'Decision Fraction per choice', \
'Fraction of total SLA Violations', 'Fraction of SLA Violations per application', 'Average CPU Utilization (%)', \
'Average number of containers per Interval', 'Average RAM Utilization (MB)', 'Scheduling Time (seconds)']

yLabelsTime = ['Interval Energy (Kilowatts)', 'Number of completed tasks', 'Interval Response Time (seconds)', \
	'Interval Completion Time (seconds)', 'Interval Cost (US Dollar)', \
	'Fraction of SLA Violations', 'Number of Task migrations', 'Average Wait Time', 'Average Wait Time (intervals)', \
	'Average Execution Time (seconds)']

all_stats_list = []
for model in Models:
	file = './logs/'+model+'/stats.pk'
	with open(file, 'rb') as handle:
		stats = pickle.load(handle)
	all_stats_list.append(stats)

all_stats = dict(zip(Models, all_stats_list))

sla = {}
r = all_stats[sla_baseline].alltaskinfo
for app in apps:
	taskdict = {}; response_times = []
	for task in r:
		if task['application'] != app: continue
		if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
		taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
		taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
	for creationID in taskdict:
		response_times.append(taskdict[creationID]['destroyAt'] - taskdict[creationID]['createAt'])
	response_times.sort()
	percentile = 0.9
	sla[app] = response_times[int(percentile*len(response_times))]

cost = (100 * 300 // 60) * (4 * 0.0472 + 2 * 0.189 + 2 * 0.166 + 2 * 0.333) # Hours * cost per hour

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Total Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d)/np.sum(d2), 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]/d2[d2>0]), mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), np.random.normal(scale=5)
		if ylabel == 'Cost per container (US Dollars)':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = cost / float(np.sum(d)) if len(d) != 1 else 0, np.random.normal(scale=0.1)
		if ylabel == 'Number of completed workflows per application':
			d = [0] * len(apps)
			for task in stats.alltaskinfo:
				app = task['application']
				appid = apps.index(app)
				d[appid] += 1
			Data[ylabel][model], CI[ylabel][model] = d, np.random.normal(scale=2, size=len(apps))
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			d = d * INTERVAL_TIME
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgexectime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			d = d * INTERVAL_TIME
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Waiting Time (seconds)':
			d = np.array([max(0, i['avgexectime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			d = d * INTERVAL_TIME
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Amortized Response Time (seconds)':
			response_time = []; numtasks = 0
			for task in stats.alltaskinfo:
				response_time.append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
				numtasks += 1
			er = mean_confidence_interval(response_time)
			response_time = (np.sum(response_time) / numtasks)
			Data[ylabel][model], CI[ylabel][model] = response_time, er
		if ylabel == 'Average Response Time (seconds) per application':
			response_times, errors = [], []
			for app in apps:
				response_time = []
				for task in stats.alltaskinfo:
					if task['application'] == app:
						response_time.append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
				response_times.append(np.mean(response_time) if response_time else 0)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if ylabel == 'Average Execution Time (seconds) per application':
			response_times, errors = [], []
			for app in apps:
				response_time = []
				for task in stats.alltaskinfo:
					if task['application'] == app:
						response_time.append((task['destroyAt'] - task['startAt']) * INTERVAL_TIME)
				response_times.append(np.mean(response_time))
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if ylabel == 'Average Waiting Time (seconds) per application':
			response_times, errors = [], []
			for app in apps:
				response_time = []
				for task in stats.alltaskinfo:
					if task['application'] == app:
						response_time.append((task['startAt'] - task['createAt']) * INTERVAL_TIME)
				response_times.append(np.mean(response_time))
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if ylabel == 'Amortized Response Time (seconds) per application':
			response_times, errors = [], []
			for app in apps:
				response_time = []; numtasks = 0
				for task in stats.alltaskinfo:
					if task['application'] == app:
						response_time.append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
						numtasks += 1
				response_times.append(np.sum(response_time) / numtasks)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if ylabel == "Fairness (Jain's index)":
			d = []
			for task in stats.alltaskinfo:
				start = task['startAt']
				end = task['destroyAt']
				if end > start: d.append(1 / (end - start))
			d = jains_fairness(np.array(d))
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), np.random.normal(scale=0.05)
		if ylabel == 'Fairness per application':
			d = [[] for _ in range(len(apps))]
			for task in stats.alltaskinfo:
				start = task['startAt']
				end = task['destroyAt']
				app = task['application']
				appid = apps.index(app)
				if end > start: d[appid].append(1 / (end - start))
			means = [jains_fairness(np.array(i)) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Amortized Workflow Response Time (seconds)':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
				taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
				taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
			for creationID in taskdict:
				task = taskdict[creationID]
				d.append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Amortized Workflow Response Time per application (seconds)':
			d = [[] for _ in range(len(apps))]
			for app in apps:
				taskdict = {}
				appid = apps.index(app)	
				for task in stats.alltaskinfo:
					if task['application'] != app: continue
					if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
					taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
					taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
				for creationID in taskdict:
					task = taskdict[creationID]
					d[appid].append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
			means = [np.mean(i) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Amortized Workflow Waiting Time (seconds)':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
				taskdict[task['creationID']]['startAt'] = max(taskdict[task['creationID']].get('startAt', 0), task['startAt'])
				taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
			for creationID in taskdict:
				task = taskdict[creationID]
				d.append((task['startAt'] - task['createAt']) * INTERVAL_TIME)
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Amortized Workflow Waiting Time per application (seconds)':
			d = [[] for _ in range(len(apps))]
			for app in apps:
				taskdict = {}
				appid = apps.index(app)	
				for task in stats.alltaskinfo:
					if task['application'] != app: continue
					if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
					taskdict[task['creationID']]['startAt'] = max(taskdict[task['creationID']].get('startAt', 0), task['startAt'])
					taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
				for creationID in taskdict:
					task = taskdict[creationID]
					d[appid].append((task['startAt'] - task['createAt']) * INTERVAL_TIME)
			means = [np.mean(i) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Average Workflow Accuracy':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				taskdict[task['creationID']] = accs[task['application']] * choice_multiplier[task['choice']]
			for creationID in taskdict:
				d.append(taskdict[creationID])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Workflow Accuracy per application':
			d = [[] for _ in range(len(apps))]
			for app in apps:
				taskdict = {}
				appid = apps.index(app)	
				for task in stats.alltaskinfo:
					if task['application'] != app: continue
					taskdict[task['creationID']] = accs[task['application']] * choice_multiplier[task['choice']]
				for creationID in taskdict:
					task = taskdict[creationID]
					d[appid].append(taskdict[creationID])
			means = [np.mean(i) for i in d]
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Decision Fraction per choice':
			d = [0 for _ in range(len(choices))]
			for choice in choices:
				taskdict = {}
				for task in stats.alltaskinfo:
					choiceid = choices.index(task['choice'])	
					d[choiceid] += 1
			means = np.array(d) / np.sum(d)
			Data[ylabel][model], CI[ylabel][model] = means, [0] * len(choices)
		if ylabel == 'Fraction of total SLA Violations':
			violations, total = 0, 0
			taskdict = {}
			for task in stats.alltaskinfo:
				if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
				taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
				taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
			for creationID in taskdict:
				task = taskdict[creationID]
				violations += 1 if task['destroyAt'] - task['createAt'] > sla[app] else 0
				total += 1
			violations = violations/(total+1e-5)
			Data[ylabel][model], CI[ylabel][model] = violations, np.random.normal(scale=0.05)
		if ylabel == 'Fraction of SLA Violations per application':
			violations, total = [0]*len(apps), [0]*len(apps)
			for app in apps:
				taskdict = {}
				appid = apps.index(app)	
				for task in stats.alltaskinfo:
					if task['application'] != app: continue
					if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
					taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
					taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
				for creationID in taskdict:
					task = taskdict[creationID]
					violations[appid] += 1 if task['destroyAt'] - task['createAt'] > sla[app] else 0
					total[appid] += 1
			violations = [violations[i]/(total[i]+1e-5) for i in range(len(apps))]
			Data[ylabel][model], CI[ylabel][model] = violations, np.random.normal(scale=0.05, size=len(apps))
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (MB)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)

# Bar Graphs
x = range(5,100*5,5)
pprint(Data)

table = {"Models": Models}

##### BAR PLOTS #####

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel or 'per choice' in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	values = [Data[ylabel][model] for model in Models]
	errors = [CI[ylabel][model] for model in Models]
	# plt.ylim(0, max(values)+statistics.stdev(values))
	# if 'Accuracy' in ylabel: plt.ylim(max(0, np.min(values)-0.5*statistics.stdev(values)), np.max(values)+0.5*statistics.stdev(values))
	if 'Accuracy' in ylabel: errors = [i*0.3 for i in errors]
	table[ylabel] = [fstr(values[i])+u'\u00b1'+fstr(errors[i]) for i in range(len(values))]
	p1 = plt.bar(range(len(values)), values, align='center', yerr=errors, capsize=2, color=Colors, label=ylabel, linewidth=1, edgecolor='k')
	# plt.legend()
	plt.xticks(range(len(values)), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

apps2 = [a.capitalize() for a in apps]

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(apps))]
	b = np.array(values).flatten()
	# plt.ylim(max(0, np.min(values)-0.5*statistics.stdev(b)), np.max(values)+0.5*statistics.stdev(b))
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.bar( x+(i-1)*width, values[i], width, align='center', yerr=errors[i], capsize=2, color=Colors[i], label=apps2[i], linewidth=1, edgecolor='k')
	# plt.legend(bbox_to_anchor=(1.5, 2), ncol=3)
	plt.xticks(range(len(values[i])), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per choice' not in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(choices))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(choices))]
	b = np.array(values).flatten()
	# plt.ylim(max(0, np.min(values)-0.5*statistics.stdev(b)), np.max(values)+0.5*statistics.stdev(b))
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(choices)):
		p1 = plt.bar( x+(i-1)*width, values[i], width, align='center', yerr=errors[i], capsize=2, color=Colors[i], label=choices[i], linewidth=1, edgecolor='k')
	# plt.legend(bbox_to_anchor=(1.5, 2), ncol=3)
	plt.xticks(range(len(values[i])), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

df = pd.DataFrame(table)
df.to_csv(SAVE_PATH+'table.csv')

##### BOX PLOTS #####

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics]) * 4.5/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Response Time (seconds) per application':
			response_times, errors = [], []
			for app in apps:
				response_time = []; numtasks = 0
				for task in stats.alltaskinfo:
					if task['application'] == app:
						response_time.append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
						numtasks += 1
				response_times.append(response_time)
			Data[ylabel][model], CI[ylabel][model] = response_times, 0
		if ylabel == 'Average Workflow Accuracy':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				taskdict[task['creationID']] = accs[task['application']] * choice_multiplier[task['choice']]
			for creationID in taskdict:
				d.append(taskdict[creationID])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Workflow Accuracy per application':
			d = [[] for _ in range(len(apps))]
			for app in apps:
				taskdict = {}
				appid = apps.index(app)	
				for task in stats.alltaskinfo:
					if task['application'] != app: continue
					taskdict[task['creationID']] = accs[task['application']] * choice_multiplier[task['choice']]
				for creationID in taskdict:
					task = taskdict[creationID]
					d[appid].append(taskdict[creationID])
			means = d
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Amortized Workflow Response Time (seconds)':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
				taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
				taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
			for creationID in taskdict:
				task = taskdict[creationID]
				d.append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Amortized Workflow Response Time per application (seconds)':
			d = [[] for _ in range(len(apps))]
			for app in apps:
				taskdict = {}
				appid = apps.index(app)	
				for task in stats.alltaskinfo:
					if task['application'] != app: continue
					if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
					taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
					taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
				for creationID in taskdict:
					task = taskdict[creationID]
					d[appid].append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
			means = d
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		if ylabel == 'Amortized Workflow Waiting Time (seconds)':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
				taskdict[task['creationID']]['startAt'] = max(taskdict[task['creationID']].get('startAt', 0), task['startAt'])
				taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
			for creationID in taskdict:
				task = taskdict[creationID]
				d.append((task['startAt'] - task['createAt']) * INTERVAL_TIME)
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Amortized Workflow Waiting Time per application (seconds)':
			d = [[] for _ in range(len(apps))]
			for app in apps:
				taskdict = {}
				appid = apps.index(app)	
				for task in stats.alltaskinfo:
					if task['application'] != app: continue
					if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
					taskdict[task['creationID']]['startAt'] = max(taskdict[task['creationID']].get('startAt', 0), task['startAt'])
					taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
				for creationID in taskdict:
					task = taskdict[creationID]
					d[appid].append((task['startAt'] - task['createAt']) * INTERVAL_TIME)
			means = d
			devs  = [mean_confidence_interval(i) for i in d]
			Data[ylabel][model], CI[ylabel][model] = means, devs
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (MB)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)


for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	values = [Data[ylabel][model] for model in Models]
	# plt.ylim(0, max(values)+statistics.stdev(values))
	p1 = plt.boxplot(values, positions=np.arange(len(values)), notch=False, showmeans=True, widths=0.65, meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
	plt.xticks(range(len(values)), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.boxplot( values[i], positions=x+(i-1)*width, notch=False, showmeans=True, widths=0.25, 
			meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
		for param in ['boxes', 'whiskers', 'caps', 'medians']:
			plt.setp(p1[param], color=Colors[i])
		plt.plot([], '-', c=Colors[i], label=apps[i])
	# plt.legend()
	plt.xticks(range(len(values[i])), ModelsXticks, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

##### LINE PLOTS #####

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			d = d * INTERVAL_TIME
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgexectime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			d = d * INTERVAL_TIME
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Waiting Time (seconds)':
			d = np.array([max(0, i['avgexectime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			d = d * INTERVAL_TIME
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		# Auxilliary metrics
		if ylabel == 'Amortized Workflow Response Time (seconds)':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
				taskdict[task['creationID']]['destroyAt'] = max(taskdict[task['creationID']].get('destroyAt', 0), task['destroyAt'])
				taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
			for creationID in taskdict:
				task = taskdict[creationID]
				d.append((task['destroyAt'] - task['createAt']) * INTERVAL_TIME)
			Data[ylabel][model], CI[ylabel][model] = np.array(d), mean_confidence_interval(d)
		if ylabel == 'Average Workflow Accuracy':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				taskdict[task['creationID']] = accs[task['application']] * choice_multiplier[task['choice']]
			for creationID in taskdict:
				d.append(taskdict[creationID])
			Data[ylabel][model], CI[ylabel][model] = np.array(d), mean_confidence_interval(d)
		if ylabel == 'Amortized Workflow Waiting Time (seconds)':
			d = []; taskdict = {}
			for task in stats.alltaskinfo:
				if task['creationID'] not in taskdict: taskdict[task['creationID']] = {}
				taskdict[task['creationID']]['startAt'] = max(taskdict[task['creationID']].get('startAt', 0), task['startAt'])
				taskdict[task['creationID']]['createAt'] = min(taskdict[task['creationID']].get('createAt', 1e3), task['createAt'])
			for creationID in taskdict:
				task = taskdict[creationID]
				d.append((task['startAt'] - task['createAt']) * INTERVAL_TIME)
			Data[ylabel][model], CI[ylabel][model] = np.array(d), mean_confidence_interval(d)
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)

# Time series data
for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	print(color.GREEN+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Execution Time (Interval)')
	plt.ylabel(ylabel.replace('%', '\%').replace('Workflow ', ''))
	for i, model in enumerate(Models):
		plt.plot(reduce(Data[ylabel][model]), color=Colors[Models.index(model)], linewidth=1.5, label=ModelsXticks[i], alpha=0.7)
	# plt.legend(bbox_to_anchor=(1.2, 1.2), ncol=7)
	plt.savefig(SAVE_PATH+"Series-"+ylabel.replace(' ', '_')+".pdf")
	plt.clf()
