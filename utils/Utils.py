import os
import logging
import json
import re
from glob import glob
from subprocess import call, run, PIPE
from .ColorUtils import *

FN_PATH = './functions/'
IPS_PATH = './serverless/datacenter/ips.json'
SAMPLE_PATH = './samples/'
DSET = list(filter(lambda k: '.md' not in k, os.listdir(SAMPLE_PATH)))

def decompose(model):
	res = decomposeHelper(model)
	print(color.BLUE+f'Using {res[0]} provisioner, {res[1]} decider, {res[2]} scheduler.'+color.ENDC)
	return res

def decomposeHelper(model):
	if model in ['UAHS', 'CAHS']:
		return [model, 'Gillis', 'GOSH']
	elif model in ['Narya']:
		return [model, 'SplitPlace', 'GOBI']
	return [model] * 3	

def printDecisionAndMigrations(decision, migrations):
	print('Decision: [', end='')
	for i, d in enumerate(decision):
		if d not in migrations: print(color.FAIL, end='')
		print(d, end='')
		if d not in migrations: print(color.ENDC, end='')
		print(', ', end='') if i != len(decision)-1 else print(']')
	print()

def printDecisions(tasklist):
	print('Decisions: [', end='')
	for i, task in enumerate(tasklist):
		print(f'({task.application}, {task.sla}, {task.choice})', end='')
		print(', ', end='') if i != len(tasklist)-1 else print(']')

def printProvisioned(hostlist):
	print('Provisioned: [', end='')
	for i, h in enumerate(hostlist):
		if not h.enable: print(color.FAIL, end='')
		print(i, end='')
		if not h.enable: print(color.ENDC, end='')
		print(',', end='') if i != len(hostlist)-1 else print(']')

def unixify(paths):
	for path in paths:
		for file in os.listdir(path):
			if '.py' in file or '.sh' in file:
				_ = os.system("bash -c \"dos2unix "+path+file+" 2&> /dev/null\"")

def delfiles(creationID, taskID=None):
  fileList = glob(f'./temp/{creationID}_**' if taskID is None else f'./temp/{creationID}_{taskID}_**')
  for filePath in fileList:
    os.remove(filePath)

def getdigit(string):
  for s in string:
   if s.isdigit():
      return int(s)

def runcmd(cmd, shell=True, pipe=True):
  data = run(cmd, shell=shell, stdout=PIPE, stderr=PIPE) if pipe else run(cmd, shell=shell)
  if pipe and 'ERROR' in data.stderr.decode():
    print(cmd)
    print(color.FAIL)
    print(data.stderr.decode())
    print(color.ENDC)
    exit()
  return data.stdout.decode() if pipe else None