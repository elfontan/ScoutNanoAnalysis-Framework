# Basic analyser for ScoutingNano

This basic setup is meant to run as an example on 2025 ScoutingNano data exploiting coffea and ScoutingNanoAODSchema.

## Setup to run coffea
```
cmsrel CMSSW_15_0_2
cd CMSSW_15_0_2/src
voms-proxy-init --voms cms --valid 168:00

source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/setup.sh

cd macros/
python3 make_2025JetsCollections_coffea.py
```