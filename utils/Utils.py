import os
import logging
import json
import re
from subprocess import call
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

def runcmd(cmd, shell=True):
  data = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if 'ERROR' in data.stderr.decode():
    print(cmd)
    print(FAIL)
    print(data.stderr.decode())
    print(ENDC)
    exit()
  return data.stdout.decode(), data.stderr.decode()