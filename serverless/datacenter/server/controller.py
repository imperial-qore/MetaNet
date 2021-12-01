import json, os
import subprocess

def gethostStat(ip):
    cmd = f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'python3 ~/agent/agent.py stats'"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout.decode())
               
def gethostDetails(ip):
    cmd = f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'python3 ~/agent/agent.py'"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout.decode())

def runFunctions(ip, funcname, input_img, output_img=None):
    ''' Runs a function and return True if successful and false otherwise '''
    cmd = f'http http://{ip}:7071/api/{funcname} @{input_img}'
    if output_img: cmd += f' > {output_img}'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if output_img:
        if os.path.getsize(output_img) == 0:
            return False
    return True