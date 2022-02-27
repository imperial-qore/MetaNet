import subprocess
from time import sleep
import json
from pprint import pprint

vmlist = ['Standard_B2s']

HEADER = '\033[1m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def getdigit(string):
  for s in string:
   if s.isdigit():
      return int(s)

def run(cmd, shell=True):
  data = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if 'ERROR' in data.stderr.decode():
    print(cmd)
    print(FAIL)
    print(data.stderr.decode())
    print(ENDC)
    exit()
  print(data.stdout.decode())
  return data.stdout.decode()

servers = []

#################

print(f'{HEADER}Azure Login{ENDC}')
run('az login')

# ##################

print(f'{HEADER}Create Azure resource group{ENDC}')
run(f'az group create --location uksouth --name SimTune')

# ##################

print(f'{HEADER}Create Azure VM{ENDC}')
for i, size in enumerate(vmlist):
  name = f'vm{i+1}'
  dat = run(f'az vm create --resource-group SimTune --name {name} --size {size} --image UbuntuLTS --ssh-key-values keys/id_rsa.pub --admin-username ansible')

# ##################

print(f'{HEADER}Wait for deployment (1 minute){ENDC}')
sleep(60)

# #################

print(f'{HEADER}Open port 7071 for all VMs{ENDC}')
for i, size in enumerate(vmlist):
  name = f'vm{i+1}'
  run(f'az vm open-port --resource-group SimTune --name {name} --port 7071')

#################

print(f'{HEADER}Install Dependencies and Deploy Functions{ENDC}')
for i, size in enumerate(vmlist):
  name = f'vm{i+1}'
  ip = run(f"az vm show -d -g SimTune -n {name} --query publicIps -o tsv").strip()
  info = {'ip': ip, 'cpu': getdigit(size), 'powermodel': 'PM'+size.split('_')[1]}
  servers.append(info)
  run(f'rsync -Pav -e "ssh -i ./keys/id_rsa" ./functions/ ansible@{ip}:/home/ansible/functions/')
  run(f'rsync -Pav -e "ssh -i ./keys/id_rsa" ./serverless/datacenter/agent/ ansible@{ip}:/home/ansible/agent/')
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'python3 -m pip install psutil'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y update'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y install python3-venv python3-pip python3-distutils python3-apt'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'wget -O ~/pkg.deb -q https://packages.microsoft.com/config/ubuntu/19.04/packages-microsoft-prod.deb'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo dpkg -i ~/pkg.deb'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y update && sudo apt-get -y install ioping sysbench azure-functions-core-tools'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo chmod +x ~/functions/funcstart.sh'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo chmod +x ~/agent/probe.sh'")
  run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'cd ~/functions && ./funcstart.sh &>/dev/null'")

#################

print(f'{HEADER}Saving VM IPs to ips.json{ENDC}')
config = {'vlan': {'uname': 'ansible', 'servers': servers}}
pprint(config)
