import os
import logging
import json
import re
from subprocess import call, run, PIPE
from .ColorUtils import *

FN_PATH = './functions/'
IPS_PATH = './serverless/datacenter/ips.json'
SAMPLE_PATH = './samples/'

def printDecisionAndMigrations(decision, migrations):
	print('Decision: [', end='')
	for i, d in enumerate(decision):
		if d not in migrations: print(color.FAIL, end='')
		print(d, end='')
		if d not in migrations: print(color.ENDC, end='')
		print(',', end='') if i != len(decision)-1 else print(']')
	print()

def unixify(paths):
	for path in paths:
		for file in os.listdir(path):
			if '.py' in file or '.sh' in file:
				_ = os.system("bash -c \"dos2unix "+path+file+" 2&> /dev/null\"")

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