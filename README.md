# SECO

Serverless Co-Design Framework.

"SecoNet: Co-Design of Serverless Edge Computing Environments"

<a href="https://gitpod.io/#https://github.com/shreshthtuli/SECO/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>

System decision making to meet system level objectives by exploiting the synergism of hardware and software through their concurrent design.

## Quick Start Guide

### Installation.

```console
# install prerequisites
sudo apt -y update && apt install -y rsync
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
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

We call the NN as SecoNet (Serverless Co-Design Network). We call the learning process as CILP (from IJCAI paper).

- We run co-simulated runs to generate gold (similar to A/B testing)
- We can not run co-simulation for each action at runtime as it takes much longer than 5 seconds (typical interval duration in serverless)

For cosim optimization parameters:

- provisioner: tradeoff between utilization ratio and cost
- decider: sla violation and accuracy
- scheduler: qos.

Baselines:

- Predict+Optimization methods: ARIMA+ACO, LSTM+ACO, Decision-NN, Semi-Direct, GRAF (use for each sub-problem).
- SOTA provisioner+decider+scheduler: UAHS+Gillis+GOSH, CAHS+Gillis+GOSH (UAHS/CAHS dont need estimates, Gillis has lower sched time), Narya+SplitPlace (Narya needs latency estimates that SplitPlace provides).
- other co-design methods: CES, HASCO, RecSim.
