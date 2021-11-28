import json
import subprocess

def gethostStat(ip):
    cmd = f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'python3 ~/agent/agent.py stats'"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout.decode())
               
def gethostDetails(ip):
    cmd = f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'python3 ~/agent/agent.py'"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout.decode())