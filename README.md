# ScoutNanoAnalysis-Framework             
This basic setup is meant to provide some examples of analysers to run on 2025 ScoutingNano data exploiting coffea and ScoutingNanoAODSchema.                                                                                               
## Setup to run coffea                                                                                                                                                                   
```
cmsrel CMSSW_15_0_2
cd CMSSW_15_0_2/src
voms-proxy-init --voms cms --valid 168:00
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc11-opt/setup.sh
cd macros                
python3 make_2025JetsCollections_coffea.py                                                                                                                                      
```   
