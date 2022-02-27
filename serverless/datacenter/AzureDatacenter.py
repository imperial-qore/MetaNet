from .Datacenter import *
from time import sleep

class AzureDatacenter(Datacenter):
    def __init__(self, mode):
        self.vmlist = ['Standard_B2s'] * 8 + ['Standard_B2ms'] * 8
        super().__init__(mode)

    def setupHosts(self):
        self.servers = []
        # Login to Azure
        print(f'{color.HEADER}Azure Login{color.ENDC}')
        runcmd(f'az login --use-device-code', pipe=False)
        # Create resource group 
        print(f'{color.HEADER}Create Azure resource group{color.ENDC}')
        runcmd(f'az group create --location uksouth --name SimTune')
        # Create VMs
        print(f'{color.HEADER}Create Azure VM{color.ENDC}')
        for i, size in enumerate(self.vmlist):
          name = f'vm{i+1}'
          dat = runcmd(f'az vm create --resource-group SimTune --name {name} --size {size} --image UbuntuLTS --ssh-key-values keys/id_rsa.pub --admin-username ansible')
        # Wait for deployment
        print(f'{color.HEADER}Wait for deployment (1 minute){color.ENDC}')
        sleep(60)
        # Open port 7071
        print(f'{color.HEADER}Open port 7071 for all VMs{color.ENDC}')
        for i, size in enumerate(self.vmlist):
          name = f'vm{i+1}'
          runcmd(f'az vm open-port --resource-group SimTune --name {name} --port 7071')
        # Install dependencies
        print(f'{color.HEADER}Install Dependencies and Deploy Functions{color.ENDC}')
        for i, size in enumerate(self.vmlist):
          name = f'vm{i+1}'
          ip = runcmd(f"az vm show -d -g SimTune -n {name} --query publicIps -o tsv").strip()
          info = {'ip': ip, 'cpu': getdigit(size), 'powermodel': 'PM'+size.split('_')[1]}
          self.servers.append(info)
          runcmd(f'rsync -Pav -e "ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa" ./functions/ ansible@{ip}:/home/ansible/functions/')
          runcmd(f'rsync -Pav -e "ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa" ./serverless/datacenter/agent/ ansible@{ip}:/home/ansible/agent/')
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y update'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y install python3-venv python3-pip python3-distutils python3-apt'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'python3 -m pip install psutil'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'wget -O ~/pkg.deb -q https://packages.microsoft.com/config/ubuntu/19.04/packages-microsoft-prod.deb'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo dpkg -i ~/pkg.deb'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo apt-get -y update && sudo apt-get -y install ioping sysbench azure-functions-core-tools'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo chmod +x ~/functions/funcstart.sh'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'sudo chmod +x ~/agent/probe.sh'")
          runcmd(f"ssh -o StrictHostKeyChecking=no -i ./keys/id_rsa ansible@{ip} 'cd ~/functions && ./funcstart.sh &>/dev/null'")
        # Save to ips.json
        print(f'{color.HEADER}Saving VM IPs to ips.json{color.ENDC}')
        config = {'servers': self.servers}
        with open(IPS_PATH, 'w') as f:
            json.dump(config, f, indent=4)

    def destroyHosts(self):
        # Delete VMs
        for i, size in enumerate(self.vmlist):
          name = f'vm{i+1}'
          dat = runcmd(f'az vm delete --resource-group SimTune --name {name}')
