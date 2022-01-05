# SciNet

Gumbel Softmax !!!

SciFramework: Serverless Co-Design Framework.

"SciNet: Co-Design in Resource Management of Distributed Computing Environments"

<a href="https://gitpod.io/#https://github.com/shreshthtuli/SciNet/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>

System decision making to meet system level objectives by exploiting the synergism of hardware and software through their concurrent design.

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

As initially, the model wont predict optimal decisions, use teacher forcing to converge. Then converge again without teacher forcing.
We call the NN as SciNet (Serverless Co-Design Network). We call the learning process as CILP (from IJCAI paper).

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

## Visualization

1. Neural Network model
2. SciNet/CILP model
3. Table: r, cost, accuracy, energy, response time, sla violations, qos (baselines + Ablations + SciNet) (Ablation: w/o trans, w/o co-design)
4. Figures: waiting time (box), cpu util/hosts active (box plots), rt per application (line), acc per application (line), decision (bars), fairness (bars), decision time (stacked bars), provisioning overhead (bars).
5. Figure (single column): Training time and test loss of each demand prediction method.
6. Table (single column): Sensitivity Analysis of gamma (r, cost, qos); xi (acc, sla, qos); zeta (e, rt, qos). 
7. RPi cluster image.
