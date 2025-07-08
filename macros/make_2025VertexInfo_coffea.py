import matplotlib.pyplot as plt
import mplhep
import hist
import glob
import pickle
from tqdm import tqdm
	
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import uproot
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, ScoutingNanoAODSchema

verbose = True

print("Everything imported...")
fname = "root://xrootd-cms.infn.it//store/data/Run2025C/ScoutingPFRun3/NANOAOD/PromptReco-v1/000/393/087/00000/af63985e-4c6c-4d27-ab51-b9eb2fd70e82.root"
events = NanoEventsFactory.from_root(
    fname+":Events",
    delayed=False,
    entry_start=0,
    entry_stop=10000,
    schemaclass=ScoutingNanoAODSchema,
    metadata={"dataset": "ScoutingPFRun3_2025C"},
).events()

print("...Everything loaded!")

print(sorted(events.fields))

## Filter by number of muons
## -------------------------
events = events[ak.num(events['ScoutingMuonVtxDisplacedVertex']) > 1]

## Define an index per muon and vSV
## --------------------------------
events["ScoutingMuonVtx"] = ak.with_field(events["ScoutingMuonVtx"], ak.local_index(events["ScoutingMuonVtx"].pt), "i")
events["ScoutingMuonVtxDisplacedVertex"] = ak.with_field(events["ScoutingMuonVtxDisplacedVertex"], ak.local_index(events["ScoutingMuonVtxDisplacedVertex"].isValidVtx), "i")

num_scouting_muonVtx_displacedVertex = ak.num(events["ScoutingMuonVtxDisplacedVertex"])
ievent_max_num_scouting_muonVtx_displacedVertex = int(np.argmax(num_scouting_muonVtx_displacedVertex))
print(f"Event: {ievent_max_num_scouting_muonVtx_displacedVertex} has the most number of scouting muonVtx, which is {num_scouting_muonVtx_displacedVertex[ievent_max_num_scouting_muonVtx_displacedVertex]}.")

vtxs     = events["ScoutingMuonVtxDisplacedVertex"]
vtxIndx  = events["ScoutingMuonVtxVtxIndx"]["vtxIndx"]
offsets  = events["ScoutingMuonVtx"]["oScoutingMuonVtxVtxIndx"]
counts   = events["ScoutingMuonVtx"]["nScoutingMuonVtxVtxIndx"]

n_muons_per_event = ak.num(events["ScoutingMuonVtx"])

# indexes for all SV
grouped_idx = ak.Array([
    [vtxIndx_evt[o : o + n] for o, n in zip(evt_offsets, evt_counts)]
    for vtxIndx_evt, evt_offsets, evt_counts in zip(vtxIndx, offsets, counts)
])

# indexes for valid SV
flat_idx_valid = ak.Array([
    [evt_vtxIndx[i] for i in range(o , o+n) if evt_vtxs[evt_vtxIndx[i]].isValidVtx]
    for evt_vtxIndx, evt_offsets, evt_counts, evt_vtxs in zip(vtxIndx, offsets, counts, vtxs)
    for o, n in zip(evt_offsets, evt_counts)
])
grouped_idx_valid = ak.unflatten(flat_idx_valid, n_muons_per_event)

#if verbose:
#    print("Showing muon indices and vertices")
#    print(ak.to_list(muon_indices[0:2]))
#    print(ak.to_list(displaced_vertices[0:2]))

## Getting only the valid vertices
## -------------------------------
displaced_vertices = ak.Array([
    [vtxs_evt[i] for i in idxs]
    for vtxs_evt, idxs_evt in zip(vtxs, grouped_idx_valid)
    for idxs in idxs_evt
])
grouped_vtx = ak.unflatten(displaced_vertices, n_muons_per_event)

best_vtxIndx_perMuon = ak.Array([
    evt_vtxIndx_valid[np.argmin(evt_vtxs_valid.chi2)] if len(evt_vtxs_valid.chi2) > 0 else -1
    for evt_vtxs_valid, evt_vtxIndx_valid in zip(displaced_vertices, flat_idx_valid)
])
grouped_bestIdx = ak.unflatten(best_vtxIndx_perMuon, n_muons_per_event)

events["ScoutingMuonVtx"] = ak.with_field(
    events["ScoutingMuonVtx"],
    grouped_idx,
    "vtxIndx"
)

events["ScoutingMuonVtx"] = ak.with_field(
    events["ScoutingMuonVtx"],
    grouped_idx_valid,
    "vtxIndxValid"
)

events["ScoutingMuonVtx"] = ak.with_field(
    events["ScoutingMuonVtx"],
    grouped_vtx,
    "displaced_vertices"
)

events["ScoutingMuonVtx"] = ak.with_field(
    events["ScoutingMuonVtx"],
    grouped_bestIdx,
    "bestVtxIndx"
)


if verbose:
    sel_event = events[ievent_max_num_scouting_muonVtx_displacedVertex]
    print(sel_event)
    print(sel_event['ScoutingMuonVtxVtxIndx'].vtxIndx)
    for i, muon in enumerate(sel_event["ScoutingMuonVtx"]):
        print(f"MuonVtx {i+1}: pt={muon['pt']:<7.3f} eta={muon['eta']:<7.3f} phi={muon['phi']:<7.3f} number of displaced vertices={muon['nScoutingMuonVtxVtxIndx']}, offset = {muon['oScoutingMuonVtxVtxIndx']}")
        print(f"{ak.to_list(muon['displaced_vertices'].x)}")
        print(f"{ak.to_list(muon['vtxIndx'])}")

    for i, sv in enumerate(sel_event["ScoutingMuonVtxDisplacedVertex"]):
        print(f"MuonVtxDisplacedVertex {i+1}: x={sv['x']:<7.3f} y={sv['y']:<7.3f}")


    # Get information of displaced vertices
    for i, muon in enumerate(sel_event["ScoutingMuonVtx"]):
        print(f"MuonVtx {i+1}: displaced vertices: x: {muon.displaced_vertices.x} y:{muon.displaced_vertices.y} z:{muon.displaced_vertices.z} chi2:{muon.displaced_vertices.chi2}")

    for i, muon in enumerate(sel_event["ScoutingMuonVtx"]):
        print(f"MuonVtx {i+1} indices valid: {muon['vtxIndxValid']}")
        print(f"MuonVtx {i+1} index best: {muon['bestVtxIndx']}")


## Pairs of muons
## --------------
all_pairs = ak.combinations(events["ScoutingMuonVtx"]["i"], 2, axis=1, fields=["i", "j"])
best_vtx = events["ScoutingMuonVtx"].bestVtxIndx
same_vtx = best_vtx[all_pairs["i"]] == best_vtx[all_pairs["j"]]
dimuon_pairs = all_pairs[same_vtx]

print(ak.to_list(all_pairs[ievent_max_num_scouting_muonVtx_displacedVertex]))
print(ak.to_list(dimuon_pairs[ievent_max_num_scouting_muonVtx_displacedVertex]))
