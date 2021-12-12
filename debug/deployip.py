import subprocess
from time import sleep
import json
from pprint import pprint

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

#################

print(f'{HEADER}Install Dependencies and Deploy Functions{ENDC}')
ip = '20.117.77.195'
run(f'rsync -Pav -e "ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa" ./functions/ ansible@{ip}:/home/ansible/functions/')
run(f'rsync -Pav -e "ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa" ./serverless/datacenter/agent/ ansible@{ip}:/home/ansible/agent/')
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'python3 -m pip install psutil'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y update'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y install python3-venv python3-pip python3-distutils python3-apt'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'wget -O ~/pkg.deb -q https://packages.microsoft.com/config/ubuntu/19.04/packages-microsoft-prod.deb'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo dpkg -i ~/pkg.deb'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y update && sudo apt-get -y install ioping sysbench azure-functions-core-tools'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo chmod +x ~/functions/funcstart.sh'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo chmod +x ~/agent/probe.sh'")
run(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'cd ~/functions && ./funcstart.sh &>/dev/null'")

run(f'http http://{ip}:7071/api/onnx @debug/babyyoda.jpg > output.jpg')