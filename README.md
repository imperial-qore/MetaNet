# MetaNet

"MetaNet: Learning to Dynamically Select the Optimal Scheduler in Cloud Computing Environments"

<a href="https://gitpod.io/#https://github.com/imperial-qore/MetaNet/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>


## Quick Start Guide

### Installation.

```console
# install prerequisites
sudo apt -y update && sudo apt install -y rsync python3-pip
pip3 install --upgrade pip
pip3 install matplotlib scikit-learn
pip3 install -r requirements.txt
export PATH=$PATH:~/.local/bin
sudo chmod 400 keys/id_rsa

# install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

Or use pre-installed gitpod.

### Quick test.

```console
# deploy a single VM and print VM IP
python3 debug/deployazure.py

# test function
http http://<public_ip>:7071/api/onnx @debug/babyyoda.jpg > output.jpg
```

# Details and motivation


## Visualization
