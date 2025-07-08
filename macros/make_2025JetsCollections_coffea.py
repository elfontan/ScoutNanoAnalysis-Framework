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
import hist
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, ScoutingNanoAODSchema

verbose = True

print("Everything imported...")
fname = "root://xrootd-cms.infn.it//store/data/Run2025C/ScoutingPFRun3/NANOAOD/PromptReco-v1/000/393/087/00000/af63985e-4c6c-4d27-ab51-b9eb2fd70e82.root"

outdir = "/eos/user/e/elfontan/www/dijetAnaRun3/"

# Load events
events = NanoEventsFactory.from_root(
    fname + ":Events",
    schemaclass=ScoutingNanoAODSchema,
    metadata={"dataset": "ScoutingPFRun3_2025C"},
    entry_start=0,
    entry_stop=10000
).events()

print("...Events loaded!")

print("--------------------------------------------------------------------------------------")
print(f"Available branches: ")
print("--------------------------------------------------------------------------------------")
print(sorted(events.fields))
print("--------------------------------------------------------------------------------------")

# ----------------------- #
# Apply basic selection   #
# ----------------------- #
#trig = events.HLT.DST_PFScouting_JetHT
#events = events[trig]

# Extract reclustered fat jets collection
jets = events["ScoutingFatPFJetRecluster"]

# Select jets with pT > 20 GeV and |eta| < 5
mask = (jets.pt > 20) & (abs(jets.eta) < 5)
selected_jets = jets[mask]
n_goodjets = ak.num(selected_jets)
sel = n_goodjets >= 2
selected_jets = selected_jets[sel]

#print(f"Events passing trigger: {ak.sum(trig)}")
print(f"Events with at least two jets: {ak.sum(sel)}")

# --------------------------------- #
# Jet and dijet kinematic variables #
# --------------------------------- #
jet1 = selected_jets[:, 0]
jet2 = selected_jets[:, 1]

#print("-------> Leading Jet Pt = ", jet1.pt)
#print(type(jet1.pt), ak.type(jet1.pt))

jet1_pt = jet1.pt.compute()
jet2_pt = jet2.pt.compute()
jet1_eta = jet1.eta.compute()
jet2_eta = jet2.eta.compute()
jet1_phi = jet1.phi.compute()
jet2_phi = jet2.phi.compute()
n_goodjets = n_goodjets[sel].compute()

mjj = np.sqrt(
    2 * jet1_pt * jet2_pt * (np.cosh(jet1_eta - jet2_eta) - np.cos(jet1_phi - jet2_phi))
)

detajj = abs(jet1_eta - jet2_eta)

# ----------------------- #
# Define and histograms   #
# ----------------------- #

hists = {
    "pt1":   hist.Hist.new.Reg(50, 0, 500, name="pt", label="Leading jet pT [GeV]").Double(),
    "pt2":   hist.Hist.new.Reg(50, 0, 500, name="pt", label="Subleading jet pT [GeV]").Double(),
    "eta1":  hist.Hist.new.Reg(50, -5, 5, name="eta", label="Leading jet eta").Double(),
    "eta2":  hist.Hist.new.Reg(50, -5, 5, name="eta", label="Subleading jet eta").Double(),
    "phi1":  hist.Hist.new.Reg(64, -3.2, 3.2, name="phi", label="Leading jet phi").Double(),
    "phi2":  hist.Hist.new.Reg(64, -3.2, 3.2, name="phi", label="Subleading jet phi").Double(),
    "njet":  hist.Hist.new.Reg(10, -0.5, 9.5, name="n", label="N good jets").Double(),
    "mjj":   hist.Hist.new.Reg(80, 0, 2000, name="mjj", label="mjj [GeV]").Double(),
    "detajj":   hist.Hist.new.Reg(50, 0, 6, name="detajj", label="DEta(jj)").Double()
}


hists["pt1"].fill(pt=ak.to_numpy(jet1_pt))
hists["pt2"].fill(pt=ak.to_numpy(jet2_pt))
hists["eta1"].fill(eta=ak.to_numpy(jet1_eta))
hists["eta2"].fill(eta=ak.to_numpy(jet2_eta))
hists["phi1"].fill(phi=ak.to_numpy(jet1_phi))
hists["phi2"].fill(phi=ak.to_numpy(jet2_phi))
hists["njet"].fill(n=ak.to_numpy(n_goodjets))
hists["mjj"].fill(mjj=ak.to_numpy(mjj))
hists["detajj"].fill(detajj=ak.to_numpy(detajj))


for key, h in hists.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    mplhep.histplot(h, ax=ax)
    mplhep.cms.label("Preliminary", data=True, com="13.6", ax=ax, loc=0, fontsize=23)
    ax.set_xlabel(h.axes[0].label, fontsize=22)
    ax.set_ylabel("Events", fontsize=22)
    ax.set_title("")  
    ax.tick_params(axis='both', which='major', labelsize=16)    
    plt.tight_layout()
    plt.savefig(f"{outdir}/{key}.png")
    plt.savefig(f"{outdir}/{key}.pdf")
    plt.close()
