###################################################################################################
#   axo_studies.py                                                                                #
#   Description: process axo-triggered events and save relevant observables in histograms         #
#   Authors: Noah Zipper, Jannicke Pearkes, Ada Collins, Elliott Kauffman, Natalie Bruhwiler,     #
#            Sabrina Giorgetti                                                                    #
###################################################################################################

###################################################################################################
# IMPORTS

# library imports
import awkward as ak
from collections import defaultdict
import dask
from dask.distributed import Client
import dask_awkward as dak
import datetime
import hist
import hist.dask as hda
import json
import numpy as np
import time
import vector
vector.register_awkward()

# coffea imports
from coffea.util import save
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import coffea.processor as processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

###################################################################################################
# PROCESSING OPTIONS

json_filename = "2024_data_filelist.json"  # name of json file containing root file paths
dataset_name = "Scouting_2024I"            # name of key within json containing dataset
has_scores = True                          # whether the files contain axo anomaly score branches
is_scouting = True                         # whether the files are scouting nanos
axo_v = "v4"                               # which axo version to use for score hists
n_files = 1                                # number of files to process (-1 for all)
coffea_step_size = 50_000                  # step size for coffea processor
coffea_files_per_batch = 1                 # files per batch for coffea processor
visualize = True                          # save task graph as pdf (only use when processing 1 or 2 files)
save_hists = True                         # save histograms using pickle
do_preprocess = True                      # create a preprocessing json file 
# which reco objects to process
reco_objects = [
    "ScoutingPFJet",
    "ScoutingElectron",
    "ScoutingMuonNoVtx",
    "ScoutingPhoton"
] 
# which l1 objects to process
l1_objects = [
    "L1Jet"
] 

# which hists to save (comment out unwanted)
hist_selection = {
    "1d_scalar": [
        "anomaly_score",                       # axol1tl anomaly score
        "l1ht",                               # ht of l1 objects
        "l1met",                              # MET of l1 objects
        "total_l1mult",                       # total l1 object multiplicity
        "total_l1pt",                         # total l1 pt
        "scoutinght",                         # ht of scouting objects
        "scoutingmet",                        # MET of scouting objects
        "total_scoutingmult",                 # total scouting object multiplicity
        "total_scoutingpt",                   # total scouting pt
    ],
    "2d_scalar": [
        "anomaly_score_l1ht",               
        "anomaly_score_l1met",             
        "anomaly_score_total_l1mult",         
        "anomaly_score_total_l1pt",
        "anomaly_score_scoutinght",
        "anomaly_score_scoutingmet",
        "anomaly_score_total_scoutingmult",   
        "anomaly_score_total_scoutingpt",
    ],
    "1d_object": [
        "n",                                   # object multiplicity
        "pt",                                  # object pt
        "pt0",                                 # leading object pt
        "pt1",                                 # subleading object pt
        "eta",                                 # object eta
        "phi",                                 # object phi
    ],
    "2d_object": [
        "anomaly_score_n",
        "anomaly_score_pt",
        "anomaly_score_eta",
        "anomaly_score_phi",
    ],
    "1d_diobject": [ 
        "m_log",                               # log axis for diobject invariant mass
        "m_low",                               # low range axis for diobject invariant mass
        "m_mid",                               # mid range axis for diobject invariant mass
        "m",                                   # full range axis for diobject invariant mass
    ],
    "2d_diobject": [
        "anomaly_score_m_log",
        "anomaly_score_m_low",
        "anomaly_score_m_mid",
        "anomaly_score_m",
    ],
    "dimuon": [
        "m_log",                               # log axis for dimuon invariant mass
        "m_low",                               # low range axis for dimuon invariant mass
        "m_mid",                               # mid range axis for dimuon invariant mass
        "m",                                   # full range axis for dimuon invariant mass
    ]
}

# which branches to save (comment out unwanted)
branch_selection = {
    "dimuon": [
        # "dimuon_mass",                            # mass of dimuon pair
        # "dimuon_pt",                              # pt of dimuon pair
        # "dimuon_eta",                             # eta of dimuon pair
        # "dimuon_phi",                             # phi of dimuon pair
        # "dimuon_pt_1",                            # pt of first muon in dimuon pair
        # "dimuon_pt_2",                            # pt of second muon in dimuon pair
        # "dimuon_eta_1",                           # eta of first muon in dimuon pair
        # "dimuon_eta_2",                           # eta of second muon in dimuon pair
        # "dimuon_phi_1",                           # phi of first muon in dimuon pair
        # "dimuon_phi_2",                           # phi of second muon in dimuon pair
    ]
}

# which triggers to save (comment out unwanted or add)
triggers = [
    # 'DST_PFScouting_AXOLoose', 
    # 'DST_PFScouting_AXONominal', 
    # 'DST_PFScouting_AXOTight', 
    # 'DST_PFScouting_AXOVLoose', 
    # 'DST_PFScouting_AXOVTight',
    # 'DST_PFScouting_CICADALoose', 
    # 'DST_PFScouting_CICADAMedium', 
    # 'DST_PFScouting_CICADATight', 
    # 'DST_PFScouting_CICADAVLoose', 
    # 'DST_PFScouting_CICADAVTight',
    # 'DST_PFScouting_DoubleMuon',
    'DST_PFScouting_JetHT',
    #'DST_PFScouting_ZeroBias'
]

###################################################################################################
# DEFINE SCHEMA
class ScoutingNanoAODSchema(NanoAODSchema):
    """ScoutingNano schema builder

    ScoutingNano is a NanoAOD format that includes Scouting objects
    """

    mixins = {
        **NanoAODSchema.mixins,
        "ScoutingPFJet": "Jet",
        "ScoutingFatJet": "Jet",
        "ScoutingMuonNoVtxDisplacedVertex": "Vertex",
        "ScoutingMuonVtxDisplacedVertex": "Vertex",
        "ScoutingElectron": "Electron",
        "ScoutingPhoton": "Photon", 
        "ScoutingMuonNoVtx": "Muon",
        "ScoutingMuonVtx": "Muon"

    }
    all_cross_references = {
        **NanoAODSchema.all_cross_references
    }
  
###################################################################################################
# HELPER FUNCTIONS FOR PROCESSOR
def find_diObjs(events_obj_coll, isL1, isScouting):
    
    objs_dict = {
        "pt": events_obj_coll.pt,
        "eta": events_obj_coll.eta,
        "phi": events_obj_coll.phi,
    }
    
    # set up four-vectors based on what kind of object we are dealing with
    if isL1:
        objs = dak.zip(
            {
                "pt": events_obj_coll.pt,
                "eta": events_obj_coll.eta,
                "phi": events_obj_coll.phi,
                "mass": dak.zeros_like(events_obj_coll.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
    elif isScouting:
        try:
            objs = dak.zip(
                {
                    "pt": events_obj_coll.pt,
                    "eta": events_obj_coll.eta,
                    "phi": events_obj_coll.phi,
                    "mass": events_obj_coll.mass,
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
        except:
            objs = dak.zip(
                {
                    "pt": events_obj_coll.pt,
                    "eta": events_obj_coll.eta,
                    "phi": events_obj_coll.phi,
                    "mass": events_obj_coll.m,
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
    else:
        objs = dak.zip({ 
            k: getattr(events_obj_coll, k) for k in ["x", "y", "z", "t"] }, 
            with_name="LorentzVector", 
            behavior=events_obj_coll.behavior, 
        )
        
    # get combinations
    diObjs = dak.combinations(objs, 2, fields=["obj1", "obj2"])
    diObj = dak.zip(
        {
            "p4": diObjs.obj1+diObjs.obj2,
        },
    )
    
    # get other characteristics
    diObj["obj1_pt"] = diObjs.obj1.pt
    diObj["obj2_pt"] = diObjs.obj2.pt
    diObj["obj1_eta"] = diObjs.obj1.eta
    diObj["obj2_eta"] = diObjs.obj2.eta
    diObj["obj1_phi"] = diObjs.obj1.phi
    diObj["obj2_phi"] = diObjs.obj2.phi
    diObj["pt"] = (diObjs.obj1+diObjs.obj2).pt
    diObj["eta"] = (diObjs.obj1+diObjs.obj2).eta
    diObj["phi"] = (diObjs.obj1+diObjs.obj2).phi
    diObj["mass"] = (diObjs.obj1+diObjs.obj2).mass
        
    return diObj

def createHist_1d(
    hist_dict, dataset_axis, trigger_axis, observable_axis, hist_name, object_axis=None 
):
    if object_axis==None:
        h = hda.hist.Hist(dataset_axis, trigger_axis, observable_axis, storage="weight", label="nEvents")
    else:
        h = hda.hist.Hist(dataset_axis, trigger_axis, object_axis, observable_axis, storage="weight", label="nEvents")
        
    hist_dict[f'{hist_name}'] = h
    
    return hist_dict

def fillHist_1d(
    hist_dict, hist_name, dataset, observable, trigger_path, observable_name, object_name=None
):
    
    kwargs = {
        observable_name: observable,
        "dataset": dataset,
        "trigger": trigger_path
    }
    
    if object_name!=None:
        kwargs["object"] = object_name
    
    hist_dict[f'{hist_name}'].fill(**kwargs)
    
    return hist_dict

def createHist_2d(
    hist_dict, dataset_axis, trigger_axis, observable1_axis, observable2_axis, hist_name, object_axis = None 
):
    if object_axis==None:
        h = hda.hist.Hist(dataset_axis, trigger_axis, observable1_axis, observable2_axis, storage="weight", label="nEvents")
    else:
        h = hda.hist.Hist(dataset_axis, trigger_axis, object_axis, observable1_axis, observable2_axis, storage="weight", label="nEvents")
        
    hist_dict[f'{hist_name}'] = h
    
    return hist_dict

def fillHist_2d(
    hist_dict, hist_name, dataset, observable1, observable2, trigger_path, observable1_name, observable2_name, object_name = None
):
    
    kwargs = {
        observable1_name: observable1,
        observable2_name: observable2,
        "dataset": dataset,
        "trigger": trigger_path
    }
    
    if object_name!=None:
        kwargs["object"] = object_name
    
    hist_dict[f'{hist_name}'].fill(**kwargs)
    
    return hist_dict
    

###################################################################################################
# DEFINE COFFEA PROCESSOR
class MakeAXOHists (processor.ProcessorABC):
    def __init__(
        self, 
        trigger_paths=[],
        hists_to_process={
            "1d_scalar": [],
            "2d_scalar": [],
            "1d_object": [],
            "2d_object": [],
            "1d_diobject": [],
            "2d_diobject": [],
            "dimuon": [],
        },
        branches_to_save={
            "dimuon": [],
        },
        has_scores=True,
        axo_version="v4",
        is_scouting=False, 
        extra_cut='', 
        thresholds=None, 
        object_dict=None
    ):
        if is_scouting:
            the_object_dict =  {'ScoutingPFJet' :      {'cut' : [('pt', 30.)], 'label' : 'j'},
                                'ScoutingElectron' : {'cut' : [('pt', 10)], 'label' : 'e'},
                                'ScoutingMuonNoVtx' :     {'cut' : [('pt', 3)], 'label' : '\mu'},
                                'ScoutingPhoton' :     {'cut' : [('pt', 10)], 'label' : '\gamma'},
                                'L1Jet' :    {'cut' : [('pt', 0.1)], 'label' : 'L1j'},
                                'L1EG' :     {'cut' : [('pt', 0.1)], 'label' : 'L1e'},
                                'L1Mu' :     {'cut' : [('pt', 0.1)], 'label' : 'L1\mu'}
                               }
        else:
            the_object_dict =  {'Jet' :      {'cut' : [('pt', 30.)], 'label' : 'j'},
                                'Electron' : {'cut' : [('pt', 10)], 'label' : 'e'},
                                'Muon' :     {'cut' : [('pt', 3)], 'label' : '\mu'},
                                'L1Jet' :    {'cut' : [('pt', 0.1)], 'label' : 'L1j'},
                                'L1EG' :     {'cut' : [('pt', 0.1)], 'label' : 'L1e'},
                                'L1Mu' :     {'cut' : [('pt', 0.1)], 'label' : 'L1\mu'}
                               }      
        self.run_dict = {
            'thresholds' : thresholds if thresholds is not None else {
                'AXOVTight_EMU'  : {'name'  : 'AXO VTight', 'score' : 25000/16},
                'AXOTight_EMU'   : {'name'  : 'AXO Tight', 'score' : 20486/16},
                'AXONominal_EMU' : {'name'  : 'AXO Nominal', 'score' : 18580/16},
                'AXOLoose_EMU'   : {'name'  : 'AXO Loose', 'score' : 17596/16},
                'AXOVLoose_EMU'  : {'name'  : 'AXO VLoose', 'score' : 15717/16},
            },
            'objects' : object_dict if object_dict is not None else the_object_dict
        }

        self.sorted_keys = sorted(
            self.run_dict['thresholds'],key=lambda i: self.run_dict['thresholds'][i]['score']
        )
        self.trigger_paths = trigger_paths
        self.has_scores = has_scores
        self.is_scouting = is_scouting
        self.extra_cut = extra_cut
        self.hists_to_process = hists_to_process
        self.branches_to_save = branches_to_save
        self.axo_version = axo_version
        
        # define axes for histograms
        self.dataset_axis = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        self.trigger_axis = hist.axis.StrCategory(
            [], growth=True, name="trigger", label="Trigger"
        )
        self.object_axis = hist.axis.StrCategory(
            [], growth=True, name="object", label="Object"
        )
        if axo_version == "v3":
            self.score_axis = hist.axis.Regular(
                200, 0, 4000, name="score", label='Anomaly Score'
            )
        else:
            self.score_axis = hist.axis.Regular(
                200, 0, 500, name="score", label='Anomaly Score'
            )
        self.mult_axis = hist.axis.Regular(
            200, 0, 201, name="mult", label=r'$N_{obj}$'
        )
        self.pt_axis = hist.axis.Regular(
            500, 0, 5000, name="pt", label=r"$p_{T}$ [GeV]"
        )
        self.eta_axis = hist.axis.Regular(
            150, -5, 5, name="eta", label=r"$\eta$"
        )
        self.phi_axis = hist.axis.Regular(
            30, -4, 4, name="phi", label=r"$\phi$"
        )
        self.met_axis = hist.axis.Regular(
            250, 0, 2500, name="met", label=r"$p^{miss}_{T} [GeV]$"
        )
        self.ht_axis = hist.axis.Regular(
            100, 0, 2000, name="ht", label=r"$H_{T}$ [GeV]"
        )
        
        # invariant mass axes
        self.minv_axis = hist.axis.Regular(
            1000, 0, 3000, name="minv", label=r"$m_{obj_{1},obj_{2}}$ [GeV]")
        self.minv_axis_log = hist.axis.Regular(
            1000, 0.01, 3000, name="minv_log", label=r"$m_{obj_{1},obj_{2}}$ [GeV]", 
            transform=hist.axis.transform.log
        )
        self.minv_axis_low = hist.axis.Regular(
            500, 0, 5, name="minv_low", label=r"$m_{obj_{1},obj_{2}}$ [GeV]"
        )
        self.minv_axis_mid = hist.axis.Regular(
            500, 50, 150, name="minv_mid", label=r"$m_{obj_{1},obj_{2}}$ [GeV]"
        )
        
    def process(self, events):
        dataset = events.metadata['dataset']
        cutflow = defaultdict(int)
        cutflow['start'] = dak.num(events.event, axis=0)
        hist_dict = {}
               
        # Saturated-Jets event cut
        events = events[dak.all(events.L1Jet.pt<1000,axis=1)]
        # Saturated-MET event cut
        events = events[dak.flatten(events.L1EtSum.pt[(events.L1EtSum.etSumType==2) 
                                                      & (events.L1EtSum.bx==0)])<1040]
        
        print("Available fields:", events.fields)
        
        # create histograms that will be filled for each trigger
        if ("l1ht" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.ht_axis, 
                "l1ht"
            )
        if ("l1met" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.met_axis, 
                "l1met"
            )
        if ("total_l1mult" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.mult_axis, 
                "total_l1mult"
            )
        if ("total_l1pt" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "total_l1pt"
            )
        if ("scoutinght" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.ht_axis, 
                "scoutinght"
            )
        if ("scoutingmet" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.met_axis, 
                "scoutingmet"
            )
        if ("total_scoutingmult" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.mult_axis, 
                "total_scoutingmult"
            )
        if ("total_scoutingpt" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "total_scoutingpt"
            )
        if ("anomaly_score" in self.hists_to_process["1d_scalar"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                "anomaly_score"
            )
        if ("anomaly_score_l1ht" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis,
                self.score_axis, 
                self.ht_axis, 
                "anomaly_score_l1ht"
            )
        if ("anomaly_score_l1met" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.met_axis, 
                "anomaly_score_l1met"
            )
        if ("anomaly_score_total_l1mult" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.mult_axis, 
                "anomaly_score_total_l1mult"
            )
        if ("anomaly_score_total_l1pt" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.pt_axis, 
                "anomaly_score_total_l1pt"
            )
        if ("anomaly_score_scoutinght" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.ht_axis, 
                "anomaly_score_scoutinght"
            )
        if ("anomaly_score_scoutingmet" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict,
                self.dataset_axis, 
                self.trigger_axis,
                self.score_axis, 
                self.met_axis, 
                "anomaly_score_scoutingmet"
            )
        if ("anomaly_score_total_scoutingmult" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.mult_axis, 
                "anomaly_score_total_scoutingmult"
            )
        if ("anomaly_score_total_scoutingpt" in self.hists_to_process["2d_scalar"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.pt_axis, 
                "anomaly_score_total_scoutingpt"
            )
        if ("n" in self.hists_to_process["1d_object"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.mult_axis, 
                'n_obj', 
                object_axis=self.object_axis
            )
        if ("pt" in self.hists_to_process["1d_object"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                'pt_obj', 
                object_axis=self.object_axis
            )
        if ("pt0" in self.hists_to_process["1d_object"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                'pt0_obj', 
                object_axis=self.object_axis
            )
        if ("pt1" in self.hists_to_process["1d_object"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                'pt1_obj', 
                object_axis=self.object_axis
            )
        if ("eta" in self.hists_to_process["1d_object"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                'eta_obj', 
                object_axis=self.object_axis
            )
        if ("phi" in self.hists_to_process["1d_object"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                'phi_obj', 
                object_axis=self.object_axis
            )
        if ("m_log" in self.hists_to_process["1d_diobject"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.minv_axis_log, 
                'm_diobj_log', 
                object_axis=self.object_axis
            )
        if ("m_low" in self.hists_to_process["1d_diobject"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.minv_axis_low, 
                'm_diobj_low', 
                object_axis=self.object_axis
            )
        if ("m_mid" in self.hists_to_process["1d_diobject"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.minv_axis_mid, 
                'm_diobj_mid', 
                object_axis=self.object_axis
            )
        if ("m" in self.hists_to_process["1d_diobject"]):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.minv_axis, 
                'm_diobj', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_n" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.mult_axis, 
                'anomaly_score_n_obj', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_pt" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.pt_axis, 
                'anomaly_score_pt_obj', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_eta" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.eta_axis, 
                'anomaly_score_eta_obj', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_phi" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.phi_axis, 
                'anomaly_score_phi_obj', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_m_log" in self.hists_to_process["2d_diobject"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.minv_axis_log, 
                'anomaly_score_m_diobj_log', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_m_low" in self.hists_to_process["2d_diobject"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.minv_axis_low, 
                'anomaly_score_m_diobj_low', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_m_mid" in self.hists_to_process["2d_diobject"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.minv_axis_mid, 
                'anomaly_score_m_diobj_mid', 
                object_axis=self.object_axis
            )
        if ("anomaly_score_m" in self.hists_to_process["2d_diobject"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.score_axis, 
                self.minv_axis, 
                'anomaly_score_m_diobj', 
                object_axis=self.object_axis
            )
        if ("eta_phi" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis, 
                'eta_phi_obj', 
                object_axis=self.object_axis
            )   
        if ("n_eta" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.mult_axis, 
                self.eta_axis, 
                'n_eta_obj', 
                object_axis=self.object_axis
            )
        if ("n_pt" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.mult_axis, 
                self.pt_axis, 
                'n_pt_obj', 
                object_axis=self.object_axis
            )
        if ("eta_pt" in self.hists_to_process["2d_object"]):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis, 
                'eta_pt_obj', 
                object_axis=self.object_axis
            )
        
        # Trigger requirement
        for trigger_path in self.trigger_paths: # loop over trigger paths
            events_trig = None
                
            # select events for current trigger
            if trigger_path == "all":
                print("all")
                events_trig = events
                
            elif trigger_path == "DoubleJet":
                print("Processing DoubleJet trigger")
                double_jet_mask = ((events.DST.PFScouting_JetHT) 
                                   & ~(events.L1.HTT255er) 
                                   & ~(events.L1.HTT360er) 
                                   & ~(events.L1.HTT400er) 
                                   & ~(events.L1.SingleJet180) 
                                   & ~(events.L1.SingleJet200))
                events_trig = events[double_jet_mask]
                print("Done processing DoubleJet")
                                
            else:
                print("other")
                trig_br = getattr(events,trigger_path.split('_')[0])
                trig_path = '_'.join(trigger_path.split('_')[1:])
                events_trig = events[getattr(trig_br,trig_path)] # select events passing trigger                         
            # save cutflow information
            cutflow[trigger_path] = dak.num(events_trig.event, axis=0)
            
            # get scalar branches (l1 objects)
            if (("l1ht" in self.hists_to_process["1d_scalar"]) or ("l1met" in self.hists_to_process["1d_scalar"])):
                l1_etsums = events_trig.L1EtSum
                if ("l1ht" in self.hists_to_process["1d_scalar"]):
                    l1_ht = l1_etsums[(events_trig.L1EtSum.etSumType==1) & (events_trig.L1EtSum.bx==0)]
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        "l1ht", 
                        dataset, 
                        dak.flatten(l1_ht.pt), 
                        trigger_path, 
                        "ht"
                    )
                if ("l1met" in self.hists_to_process["1d_scalar"]):
                    l1_met = l1_etsums[(events_trig.L1EtSum.etSumType==2) & (events_trig.L1EtSum.bx==0)]
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        "l1met", 
                        dataset, 
                        dak.flatten(l1_met.pt), 
                        trigger_path, 
                        "met"
                    )
            if ("total_l1mult" in self.hists_to_process["1d_scalar"]):
                l1_total_mult = (dak.num(events_trig.L1Jet.bx[events_trig.L1Jet.bx == 0]) 
                                 + dak.num(events_trig.L1Mu.bx[events_trig.L1Mu.bx == 0]) 
                                 + dak.num(events_trig.L1EG.bx[events_trig.L1EG.bx ==0]))
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "total_l1mult", 
                    dataset, 
                    l1_total_mult, 
                    trigger_path, 
                    "mult"
                )
            if ("total_l1pt" in self.hists_to_process["1d_scalar"]):
                l1_total_pt = (dak.sum(events_trig.L1Jet.pt[events_trig.L1Jet.bx == 0],axis=1) 
                               + dak.sum(events_trig.L1Mu.pt[events_trig.L1Mu.bx == 0],axis=1) 
                               + dak.sum(events_trig.L1EG.pt[events_trig.L1EG.bx ==0],axis=1))
                hist_dict = fillHist_1d(
                    hist_dict,
                    "total_l1pt", 
                    dataset, 
                    l1_total_pt, 
                    trigger_path, 
                    "pt"
                )
                
            # get scalar branches (scouting objects)
            if ("scoutinght" in self.hists_to_process["1d_scalar"]):
                scouting_ht = dak.sum(events_trig.ScoutingPFJet.pt,axis=1)
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "scoutinght", 
                    dataset, 
                    scouting_ht, 
                    trigger_path, 
                    "ht"
                )
                
            if ("scoutingmet" in self.hists_to_process["1d_scalar"]):
                scouting_met = events_trig.ScoutingMET.pt
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "scoutingmet", 
                    dataset, 
                    scouting_met, 
                    trigger_path, 
                    "met"
                )
            if ("total_scoutingmult" in self.hists_to_process["1d_scalar"]):
                scouting_total_mult = dak.num(events_trig.ScoutingPFJet) + dak.num(events_trig.ScoutingElectron) + dak.num(events_trig.ScoutingMuonNoVtx)
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "total_scoutingmult", 
                    dataset, 
                    scouting_total_mult, 
                    trigger_path, 
                    "mult"
                )
            if ("total_scoutingpt" in self.hists_to_process["1d_scalar"]):
                scouting_total_pt = dak.sum(events_trig.ScoutingPFJet.pt,axis=1) + dak.sum(events_trig.ScoutingElectron.pt,axis=1) + dak.sum(events_trig.ScoutingMuonNoVtx.pt,axis=1)
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "total_scoutingpt", 
                    dataset, 
                    scouting_total_pt, 
                    trigger_path, 
                    "pt"
                )
            
            # score hists with scalars
            if self.has_scores:
                if self.axo_version == "v4":
                    axo_score = events_trig.axol1tl.score_v4
                elif self.axo_version == "v3":
                    axo_score = events_trig.axol1tl.score_v3
                    
                # 1d score hist
                if ("anomaly_score" in self.hists_to_process["1d_scalar"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        "anomaly_score", 
                        dataset, 
                        axo_score, 
                        trigger_path, 
                        "score"
                    )
                
                # 2d score hists with l1
                if ("anomaly_score_l1ht" in self.hists_to_process["2d_scalar"]):
                    l1_etsums = events_trig.L1EtSum
                    l1_ht = l1_etsums[(events_trig.L1EtSum.etSumType==1) & (events_trig.L1EtSum.bx==0)]
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        "anomaly_score_l1ht", 
                        dataset, 
                        dak.flatten(dak.broadcast_arrays(axo_score,l1_ht.pt)[0]),                 
                        dak.flatten(l1_ht.pt), 
                        trigger_path, 
                        "score", 
                        "ht"
                    )
                if ("anomaly_score_l1met" in self.hists_to_process["2d_scalar"]):
                    l1_etsums = events_trig.L1EtSum
                    l1_met = l1_etsums[(events_trig.L1EtSum.etSumType==2) & (events_trig.L1EtSum.bx==0)]
                    hist_dict = fillHist_2d(
                        hist_dict,
                        "anomaly_score_l1met",
                        dataset,
                        dak.flatten(dak.broadcast_arrays(axo_score,l1_met.pt)[0]), 
                        dak.flatten(l1_met.pt), 
                        trigger_path, 
                        "score", 
                        "met"
                    )
                if ("anomaly_score_total_l1mult" in self.hists_to_process["2d_scalar"]):
                    l1_total_mult = (dak.num(events_trig.L1Jet.bx[events_trig.L1Jet.bx == 0]) 
                                     + dak.num(events_trig.L1Mu.bx[events_trig.L1Mu.bx == 0]) 
                                     + dak.num(events_trig.L1EG.bx[events_trig.L1EG.bx ==0]))
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        "anomaly_score_total_l1mult", 
                        dataset, 
                        axo_score, 
                        l1_total_mult, 
                        trigger_path, 
                        "score", 
                        "mult"
                    )
                if ("anomaly_score_total_l1pt" in self.hists_to_process["2d_scalar"]):
                    l1_total_pt = (dak.sum(events_trig.L1Jet.pt[events_trig.L1Jet.bx == 0],axis=1) 
                                   + dak.sum(events_trig.L1Mu.pt[events_trig.L1Mu.bx == 0],axis=1) 
                                   + dak.sum(events_trig.L1EG.pt[events_trig.L1EG.bx ==0],axis=1))
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        "anomaly_score_total_l1pt", 
                        dataset, 
                        axo_score, 
                        l1_total_pt, 
                        trigger_path, 
                        "score", 
                        "pt"
                    )
                if ("anomaly_score_scoutinght" in self.hists_to_process["2d_scalar"]):
                    scouting_ht = dak.sum(events_trig.ScoutingPFJet.pt,axis=1)
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        "anomaly_score_scoutinght", 
                        dataset, 
                        axo_score, 
                        scouting_ht, 
                        trigger_path, 
                        "score", 
                        "ht"
                    )
                if ("anomaly_score_scoutingmet" in self.hists_to_process["2d_scalar"]):
                    scouting_met = events_trig.ScoutingMET.pt
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        "anomaly_score_scoutingmet", 
                        dataset, 
                        axo_score,
                        scouting_met, 
                        trigger_path, 
                        "score", 
                        "met"
                    )
                if ("anomaly_score_total_scoutingmult" in self.hists_to_process["2d_scalar"]):
                    scouting_total_mult = dak.num(events_trig.ScoutingPFJet) + dak.num(events_trig.ScoutingElectron) + dak.num(events_trig.ScoutingMuonNoVtx)
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        "anomaly_score_total_scoutingmult", 
                        dataset, 
                        axo_score,            
                        scouting_total_mult, 
                        trigger_path, 
                        "score", 
                        "mult"
                    )
                if ("anomaly_score_total_scoutingpt" in self.hists_to_process["2d_scalar"]):
                    scouting_total_pt = dak.sum(events_trig.ScoutingPFJet.pt,axis=1) + dak.sum(events_trig.ScoutingElectron.pt,axis=1) + dak.sum(events_trig.ScoutingMuonNoVtx.pt,axis=1)
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        "anomaly_score_total_scoutingpt", 
                        dataset, 
                        axo_score, 
                        scouting_total_pt, 
                        trigger_path, 
                        "score", 
                        "pt"
                    )
                    
            # Process object collections - w/trigger
            for obj,obj_dict in self.run_dict['objects'].items():
                cut_list = obj_dict['cut']
                label = obj_dict['label']
                isL1Obj = 'L1' in obj
                isScoutingObj = 'Scouting' in obj
                br = getattr(events_trig, obj)
                
                # Filter only L1 Objects from BX==0
                if isL1Obj:
                    br = br[br.bx==0]

                # Apply list of cuts to relevant branches
                for var, cut in cut_list:
                    mask = (getattr(br,var) > cut)
                    br = br[mask]

                # Build di-object candidate
                objs = br[dak.argsort(br.pt, axis=1)]
                
                # Fill 1D object histograms - w/trigger
                if ("n" in self.hists_to_process["1d_object"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'n_obj', 
                        dataset, 
                        dak.num(br), 
                        trigger_path, 
                        "mult", 
                        object_name=obj
                    )
                if ("pt" in self.hists_to_process["1d_object"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'pt_obj', 
                        dataset, 
                        dak.flatten(br.pt), 
                        trigger_path, 
                        "pt", 
                        object_name=obj
                    )
                if ("pt0" in self.hists_to_process["1d_object"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'pt0_obj', 
                        dataset, 
                        dak.flatten(br.pt[:,0:1]), 
                        trigger_path, 
                        "pt", 
                        object_name=obj
                    )
                if ("pt1" in self.hists_to_process["1d_object"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'pt1_obj', 
                        dataset, 
                        dak.flatten(br.pt[:,1:2]), 
                        trigger_path, 
                        "pt", 
                        object_name=obj
                    )
                if ("eta" in self.hists_to_process["1d_object"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'eta_obj', 
                        dataset, 
                        dak.flatten(br.eta), 
                        trigger_path, 
                        "eta", 
                        object_name=obj
                    )
                if ("phi" in self.hists_to_process["1d_object"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'phi_obj', 
                        dataset, 
                        dak.flatten(br.phi), 
                        trigger_path, 
                        "phi", 
                        object_name=obj
                    )
                
                diObj = find_diObjs(objs[:,0:2], isL1Obj,isScoutingObj)
                        
                # Fill 1D diobject histograms - w/trigger
                if ("m_log" in self.hists_to_process["1d_diobject"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'm_diobj_log', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv_log", 
                        object_name=obj
                    )
                if ("m_low" in self.hists_to_process["1d_diobject"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'm_diobj_low', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv_low", 
                        object_name=obj
                    )
                if ("m_mid" in self.hists_to_process["1d_diobject"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'm_diobj_mid', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv_mid", 
                        object_name=obj
                    )
                if ("m" in self.hists_to_process["1d_diobject"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'm_diobj', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv", 
                        object_name=obj
                    )
                
                # Fill 2D histograms - w/trigger
                if self.has_scores:
                    if ("anomaly_score_n" in self.hists_to_process["2d_object"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_n_obj', 
                            dataset, 
                            axo_score, 
                            dak.num(br), 
                            trigger_path, 
                            "score", 
                            "mult", 
                            object_name=obj
                        )
                    if ("anomaly_score_pt" in self.hists_to_process["2d_object"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_pt_obj', 
                            dataset, 
                            dak.flatten(dak.broadcast_arrays(axo_score,br.pt)[0]), 
                            dak.flatten(br.pt), 
                            trigger_path, 
                            "score", 
                            "pt", 
                            object_name=obj
                        )
                    if ("anomaly_score_eta" in self.hists_to_process["2d_object"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_eta_obj', 
                            dataset, 
                            dak.flatten(dak.broadcast_arrays(axo_score,br.eta)[0]), 
                            dak.flatten(br.eta), 
                            trigger_path, 
                            "score", 
                            "eta", 
                            object_name=obj
                        )
                    if ("anomaly_score_phi" in self.hists_to_process["2d_object"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_phi_obj', 
                            dataset, 
                            dak.flatten(dak.broadcast_arrays(axo_score,br.phi)[0]), 
                            dak.flatten(br.phi), 
                            trigger_path, 
                            "score", 
                            "phi", 
                            object_name=obj
                        )
                    if ("anomaly_score_m_log" in self.hists_to_process["2d_diobject"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_m_diobj_log', 
                            dataset, 
                            dak.flatten(dak.broadcast_arrays(axo_score,diObj.mass)[0]), 
                            dak.flatten(diObj.mass), 
                            trigger_path, 
                            "score", 
                            "minv_log",
                            object_name=obj
                        )
                    if ("anomaly_score_m_low" in self.hists_to_process["2d_diobject"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_m_diobj_low', 
                            dataset, 
                            dak.flatten(dak.broadcast_arrays(axo_score,diObj.mass)[0]), 
                            dak.flatten(diObj.mass), 
                            trigger_path, 
                            "score", 
                            "minv_low", 
                            object_name=obj
                        )
                    if ("anomaly_score_m_mid" in self.hists_to_process["2d_diobject"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_m_diobj_mid', 
                            dataset, 
                            dak.flatten(dak.broadcast_arrays(axo_score,diObj.mass)[0]), 
                            dak.flatten(diObj.mass), 
                            trigger_path, 
                            "score", 
                            "minv_mid", 
                            object_name=obj
                        )
                    if ("anomaly_score_m" in self.hists_to_process["2d_diobject"]):
                        hist_dict = fillHist_2d(
                            hist_dict, 
                            'anomaly_score_m_diobj', 
                            dataset, 
                            dak.flatten(dak.broadcast_arrays(axo_score,diObj.mass)[0]), 
                            dak.flatten(diObj.mass), 
                            trigger_path, 
                            "score", 
                            "minv", 
                            object_name=obj
                        )
                
                if ("eta_phi" in self.hists_to_process["2d_object"]):
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        'eta_phi_obj', 
                        dataset, 
                        dak.flatten(br.eta),
                        dak.flatten(br.phi), 
                        trigger_path, 
                        "eta", 
                        "phi", 
                        object_name=obj
                    )
                if ("n_eta" in self.hists_to_process["2d_object"]):
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        'n_eta_obj', 
                        dataset, 
                        dak.flatten(dak.broadcast_arrays(dak.num(br),br.eta)[0]), 
                        dak.flatten(br.eta), 
                        trigger_path, 
                        "mult", 
                        "eta", 
                        object_name=obj
                    )
                if ("n_pt" in self.hists_to_process["2d_object"]):
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        'n_pt_obj', 
                        dataset, 
                        dak.flatten(dak.broadcast_arrays(dak.num(br),br.pt)[0]), 
                        dak.flatten(br.pt), 
                        trigger_path, 
                        "mult", 
                        "pt", 
                        object_name=obj
                    )
                if ("eta_pt" in self.hists_to_process["2d_object"]):
                    hist_dict = fillHist_2d(
                        hist_dict, 
                        'eta_pt_obj', 
                        dataset, 
                        dak.flatten(br.eta), 
                        dak.flatten(br.pt), 
                        trigger_path, 
                        "eta", 
                        "pt", 
                        object_name=obj
                    )
                    
        # dimuon analysis
        if len(self.hists_to_process["dimuon"])>0:
            # At least two opposite sign muons
            events = events[(dak.num(events.ScoutingMuonNoVtx,axis=1)>=2) & (dak.sum(events.ScoutingMuonNoVtx.charge[:,0:2],axis=1)==0)]
            obj = "ScoutingMuonNoVtx"
            obj_dict = self.run_dict['objects'][obj]
            
            # save branches if enabled
            branch_save_dict = {}
            
            # create histograms to fill for each trigger
            if ("m_log" in self.hists_to_process["dimuon"]):
                hist_dict = createHist_1d(
                    hist_dict, 
                    self.dataset_axis, 
                    self.trigger_axis, 
                    self.minv_axis_log, 
                    'dimuon_m_log'
                )
            if ("m_low" in self.hists_to_process["dimuon"]):
                hist_dict = createHist_1d(
                    hist_dict, 
                    self.dataset_axis, 
                    self.trigger_axis, 
                    self.minv_axis_low, 
                    'dimuon_m_low'
                )
            if ("m_mid" in self.hists_to_process["dimuon"]):
                hist_dict = createHist_1d(
                    hist_dict, 
                    self.dataset_axis, 
                    self.trigger_axis, 
                    self.minv_axis_mid, 
                    'dimuon_m_mid'
                )
            if ("m" in self.hists_to_process["dimuon"]):
                hist_dict = createHist_1d(
                    hist_dict, 
                    self.dataset_axis, 
                    self.trigger_axis, 
                    self.minv_axis, 
                    'dimuon_m'
                )

            for trigger_path in self.trigger_paths: # loop over trigger paths
                events_trig = None

                if trigger_path == "all":
                    events_trig = events
                else:
                    trig_br = getattr(events,trigger_path.split('_')[0])
                    trig_path = '_'.join(trigger_path.split('_')[1:])
                    events_trig = events[getattr(trig_br,trig_path)] # select events passing this trigger
                cutflow["dimuon"+trigger_path] = dak.num(events_trig.event, axis=0)

                cut_list = obj_dict['cut']
                label = obj_dict['label']
                isL1Obj = 'L1' in obj
                isScoutingObj = 'Scouting' in obj
                br = getattr(events_trig, obj)

                # Apply list of cuts to relevant branches
                for var, cut in cut_list:
                    mask = (getattr(br,var) > cut)
                    br = br[mask]        

                # Build di-object candidate
                objs = br[dak.argsort(br.pt, axis=1)]
                diObj = find_diObjs(objs[:,0:2], isL1Obj,isScoutingObj)

                if ("m_log" in self.hists_to_process["dimuon"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'dimuon_m_log', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv_log"
                    )
                if ("m_low" in self.hists_to_process["dimuon"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'dimuon_m_low', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv_low"
                    )
                if ("m_mid" in self.hists_to_process["dimuon"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'dimuon_m_mid', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv_mid"
                    )
                if ("m" in self.hists_to_process["dimuon"]):
                    hist_dict = fillHist_1d(
                        hist_dict, 
                        'dimuon_m', 
                        dataset, 
                        dak.flatten(diObj.mass), 
                        trigger_path, 
                        "minv"
                    )
                    
                # save branches if enabled
                if "dimuon_mass" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_mass_{trigger_path}"] = dak.flatten(diObj.mass)
                if "dimuon_pt" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_pt_{trigger_path}"] = dak.flatten(diObj.pt)
                if "dimuon_eta" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_eta_{trigger_path}"] = dak.flatten(diObj.eta)
                if "dimuon_phi" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_phi_{trigger_path}"] = dak.flatten(diObj.phi)
                if "dimuon_pt_1" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_pt_1_{trigger_path}"] = dak.flatten(diObj.obj1_pt)
                if "dimuon_pt_2" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_pt_2_{trigger_path}"] = dak.flatten(diObj.obj2_pt)
                if "dimuon_eta_2" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_eta_1_{trigger_path}"] = dak.flatten(diObj.obj1_eta)
                if "dimuon_eta_2" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_eta_2_{trigger_path}"] = dak.flatten(diObj.obj2_eta)
                if "dimuon_phi_1" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_phi_1_{trigger_path}"] = dak.flatten(diObj.obj1_phi)
                if "dimuon_phi_2" in self.branches_to_save["dimuon"]: 
                    branch_save_dict[f"dimuon_phi_2_{trigger_path}"] = dak.flatten(diObj.obj2_phi)
                   
            if len(self.branches_to_save["dimuon"])>0:
                dak_zip = dak.zip(branch_save_dict)
                dak_zip.persist().to_parquet("branches")
            
        return_dict = {}
        return_dict['cutflow'] = cutflow
        return_dict['hists'] = hist_dict
        return_dict['trigger'] = self.trigger_paths if len(self.trigger_paths)>0 else None
                
        return return_dict

    def postprocess(self, accumulator):
        return accumulator


###################################################################################################
# DEFINE MAIN FUNCTION
def main():
    todaysDate = datetime.date.today().strftime('%Y%m%d')
    
    client = Client("tls://localhost:8786")
    
    with open(json_filename) as json_file:
        dataset = json.load(json_file)
    
    dataset_skimmed = {dataset_name: {'files': {}}}
    i = 0
    for key, value in dataset[dataset_name]['files'].items():
        if ((i<n_files) or (n_files==-1)):
            dataset_skimmed[dataset_name]['files'][key] = value
            i+=1
         
    print(f"Processing {i} files")
    tstart = time.time()
    if do_preprocess:
        dataset_runnable, dataset_updated = preprocess(
            dataset_skimmed,
            align_clusters=False,
            step_size=coffea_step_size,
            files_per_batch=coffea_files_per_batch,
            skip_bad_files=True,
            save_form=False,
        )
        with open('preprocessed.json', 'w') as f:
            json.dump((dataset_runnable, dataset_updated), f)
        print("Saved preprocessed filelist as preprocessed.json")
        
        print(f'{time.time()-tstart:.1f}s to pre-process')
    else: 
        with open('preprocessed.json', 'r') as f:
            print("Loading preprocessed filelist from preprocessed.json")
            dataset_runnable, dataset_updated = json.load(f)
    

    tstart = time.time()
    
    to_compute = apply_to_fileset(
        MakeAXOHists(trigger_paths=triggers, 
                     hists_to_process=hist_selection,
                     branches_to_save=branch_selection,
                     has_scores=has_scores, 
                     axo_version=axo_v,
                     is_scouting=is_scouting),
        max_chunks(dataset_runnable, 300000),
        schemaclass=ScoutingNanoAODSchema,
        uproot_options={"allow_read_errors_with_report": (OSError, TypeError, KeyError)}
    )
    
    if visualize:
        dask.optimize(to_compute)
        dask.visualize(to_compute, filename=f"dask_coffea_graph_{todaysDate}", format="pdf")

    (hist_result,) = dask.compute(to_compute)
    print(f'{time.time()-tstart:.1f}s to process')
    hist_result = hist_result[0]

    #Save file 
    if save_hists:
        save(hist_result, f'hist_result_{dataset_name}_{todaysDate}.pkl')
    

###################################################################################################
# RUN SCRIPT
if __name__=="__main__":
    main()
