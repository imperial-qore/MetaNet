import json
import psutil
import time
import subprocess
import os
import sys

def hostDetails():
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    data = subprocess.run("~/scripts/probe.sh", shell=True,stdout=subprocess.PIPE)
    data  = (data.stdout.decode()).splitlines()
    bw = ((subprocess.run("sudo ethtool "+self.interface+" | grep Speed",shell=True,stdout=subprocess.PIPE)).stdout.decode()).split()[1][0:4]
    payload ={
            "Total_Memory": int(float(memory.total/(1024*1024))),
            "Total_Disk": int(float(disk.total/(1024*1024))),
            "Bandwidth": int(bw),
            "clock": data[0],
            "Ram_read": self.parse_io(data[3]),
            "Ram_write": self.parse_io(data[4]),
            "Disk_read": self.parse_io(data[1]),
            "Disk_write": self.parse_io(data[2])}
     return json.dumps(payload)

def gethostStat():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()[2]
    disk = psutil.disk_usage('/')
    disk_total = disk.used / (1024 * 1024)
    ts = time.time()
    payload = {"time-stamp":ts, "cpu":cpu, "memory":memory, "disk":disk_total}
    print(json.dumps(payload))

if __name__ == '__main__':
    if sys.argv[-1] == 'stats':
        print(gethostStat())
    else:
        print(hostDetails())