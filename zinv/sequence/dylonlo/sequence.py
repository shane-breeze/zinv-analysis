from alphatwirl.loop import NullCollector
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
toppath = os.path.abspath(os.path.join(os.environ["TOPDIR"], "zinv"))
datapath = os.path.join(toppath, "data")
collpath = os.path.join(toppath, "sequence", "Collectors")

import zinv.sequence.Readers as Readers
import zinv.sequence.Collectors as Collectors

event_tools = Readers.EventTools(
    name = "event_tools",
    maxsize = int(2*1024**3), # 2 GB
)

# Initialise readers and collectors
collection_creator = Readers.CollectionCreator(
    name = "collection_creator",
    collections = ["CaloMET", "MET", "Jet", "Electron", "Muon", "Photon", "Tau",
                   "GenMET", "GenPart", "GenJet", "GenDressedLepton", "LHEPart"],
)

# Gen/Lhe level producers
gen_boson_producer = Readers.GenBosonProducer(
    name = "gen_boson_producer",
    data = False,
)
lhe_part_assigner = Readers.LHEPartAssigner(
    name = "lhe_part_assigner",
    old_parents = ["WJetsToLNu", "DYJetsToLL", "ZJetsToLL", "GStarJetsToLL"],
    data = False,
)
gen_part_assigner = Readers.GenPartAssigner(
    name = "gen_part_assigner",
    old_parents = ["WJetsToLNu", "DYJetsToLL", "ZJetsToLL", "GStarJetsToLL"],
    data = False,
)

jec_variations = Readers.JecVariations(
    name = "jec_variations",
    jes_unc_file = datapath + "/jecs/Summer16_23Sep2016V4_MC_UncertaintySources_AK4PFchs.txt",
    jer_sf_file = datapath + "/jecs/Spring16_25nsV10a_MC_SF_AK4PFchs.txt",
    jer_file = datapath + "/jecs/Spring16_25nsV10_MC_PtResolution_AK4PFchs.txt",
    apply_jer_corrections = True,
    jes_regex = "jes(?P<source>.*)",
    unclust_threshold = 15.,
    maxdr_jets_with_genjets = 0.2,
    ndpt_jets_with_genjets = 3.,
    data = False,
)

object_functions = Readers.ObjectFunctions(
    name = "object_functions",
    unclust_threshold = 15.,
    selections = [
        ("Jet", "JetVeto", True),
        ("Jet", "JetVetoNoSelection", True),
        ("Jet", "JetSelection", True),
        ("Jet", "JetFwdSelection", True),
        ("Jet", "JetBVeto", True),
        ("Jet", "JetBVetoNoSelection", True),
        ("Jet", "JetBSelection", True),
        ("Muon", "MuonVeto", False),
        ("Muon", "MuonVetoNoSelection", False),
        ("Muon", "MuonSelection", False),
        ("Electron", "ElectronVeto", False),
        ("Electron", "ElectronVetoNoSelection", False),
        ("Electron", "ElectronSelection", False),
        ("Photon", "PhotonVeto", False),
        ("Photon", "PhotonVetoNoSelection", False),
        ("Photon", "PhotonSelection", False),
        ("Tau", "TauVeto", True),
        ("Tau", "TauVetoNoSelection", True),
        ("Tau", "TauSelection", True),
    ],
)

event_functions = Readers.EventFunctions(
    name = "event_functions",
)

skim_collections = Readers.SkimCollections(
    name = "skim_collections",
)

tau_cross_cleaning = Readers.ObjectCrossCleaning(
    name = "tau_cross_cleaning",
    collections = ("Tau",),
    ref_collections = ("MuonVeto", "ElectronVeto"),
    mindr = 0.4,
)

jet_cross_cleaning = Readers.ObjectCrossCleaning(
    name = "jet_cross_cleaning",
    collections = ("Jet",),
    ref_collections = ("MuonVeto", "ElectronVeto", "PhotonVeto", "TauVeto"),
    mindr = 0.4,
)

trigger_checker = Readers.TriggerChecker(
    name = "trigger_checker",
)

certified_lumi_checker = Readers.CertifiedLumiChecker(
    name = "certified_lumi_checker",
    lumi_json_path = datapath + "/json/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt",
    mc = False,
)

weight_xsection_lumi = Readers.WeightXsLumi(
    name = "weight_xsection_lumi",
    data = False,
)
weight_pdf_scale = Readers.WeightPdfScale(
    name = "weight_pdf_scale",
    #parents_to_skip = ["SingleTop", "QCD", "Diboson"],
    data = False,
)
weight_pu = Readers.WeightPileup(
    name = "weight_pu",
    correction_file = datapath + "/pileup/nTrueInt_corrections.txt",
    variable = "Pileup_nTrueInt",
    data = False,
)
weight_met_trigger = Readers.WeightMetTrigger(
    name = "weight_met_trigger",
    correction_files = {
        0: datapath + "/mettrigger/met_trigger_correction_0mu.txt",
        1: datapath + "/mettrigger/met_trigger_correction_1mu.txt",
        2: datapath + "/mettrigger/met_trigger_correction_2mu.txt",
    },
    data = False,
)
weight_electrons = Readers.WeightObjects(
    name = "weight_electrons",
    correctors = [
        {
            "name": "eleIdIsoTight",
            "collection": "Electron",
            "binning_variables": ("ev, source, nsig: ev.Electron.eta", "ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_idiso_tight.txt")],
            "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Electron.eta)",
            "nuisances": ["eleIdIsoTight", "eleEnergyScale"],
        }, {
            "name": "eleIdIsoVeto",
            "collection": "Electron",
            "binning_variables": ("ev, source, nsig: ev.Electron.eta", "ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_idiso_veto.txt")],
            "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Electron.eta)",
            "nuisances": ["eleIdIsoVeto", "eleEnergyScale"],
        }, {
            "name": "eleReco",
            "collection": "Electron",
            "binning_variables": ("ev, source, nsig: ev.Electron.eta", "ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_reconstruction.txt")],
            "add_syst": "ev, source, nsig: 0.01*((ev.Electron_ptShift(ev, source, nsig)<20) | (ev.Electron_ptShift(ev, source, nsig)>80))",
            "nuisances": ["eleReco", "eleEnergyScale"],
        }, {
            "name": "eleTrig",
            "collection": "Electron",
            "binning_variables": ("ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)", "ev, source, nsig: np.abs(ev.Electron.eta)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_trigger_v2.txt")],
            "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Electron.eta)",
            "nuisances": ["eleTrig", "eleEnergyScale"],
        },
    ],
    data = False,
)
weight_muons = Readers.WeightObjects(
    name = "weight_muons",
    correctors = [
        {
            "name": "muonIdTight",
            "collection": "Muon",
            "binning_variables": ("ev, source, nsig: np.abs(ev.Muon.eta)", "ev, source, nsig: ev.Muon_ptShift(ev, source, nsig)"),
            "weighted_paths": [
                (19.7, datapath+"/muons/muon_id_loose_runBCDEF.txt"),
                (16.2, datapath+"/muons/muon_id_loose_runGH.txt"),
            ],
            "add_syst": "ev, source, nsig: 0.01*awk.JaggedArray.ones_like(ev.Muon.eta)",
            "nuisances": ["muonIdTight", "muonPtScale"],
        }, {
            "name": "muonIdLoose",
            "collection": "Muon",
            "binning_variables": ("ev, source, nsig: np.abs(ev.Muon.eta)", "ev, source, nsig: ev.Muon_ptShift(ev, source, nsig)"),
            "weighted_paths": [
                (19.7, datapath+"/muons/muon_id_loose_runBCDEF.txt"),
                (16.2, datapath+"/muons/muon_id_loose_runGH.txt"),
            ],
            "add_syst": "ev, source, nsig: 0.01*awk.JaggedArray.ones_like(ev.Muon.eta)",
            "nuisances": ["muonIdLoose", "muonPtScale"],
        }, {
            "name": "muonIsoTight",
            "collection": "Muon",
            "binning_variables": ("ev, source, nsig: np.abs(ev.Muon.eta)", "ev, source, nsig: ev.Muon_ptShift(ev, source, nsig)"),
            "weighted_paths": [
                (19.7, datapath+"/muons/muon_iso_tight_tightID_runBCDEF.txt"),
                (16.2, datapath+"/muons/muon_iso_tight_tightID_runGH.txt"),
            ],
            "add_syst": "ev, source, nsig: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
            "nuisances": ["muonIsoTight", "muonPtScale"],
        }, {
            "name": "muonIsoLoose",
            "collection": "Muon",
            "binning_variables": ("ev, source, nsig: np.abs(ev.Muon.eta)", "ev, source, nsig: ev.Muon_ptShift(ev, source, nsig)"),
            "weighted_paths": [
                (19.7, datapath+"/muons/muon_iso_loose_looseID_runBCDEF.txt"),
                (16.2, datapath+"/muons/muon_iso_loose_looseID_runGH.txt"),
            ],
            "add_syst": "ev, source, nsig: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
            "nuisances": ["muonIsoLoose", "muonPtScale"],
        }, {
            "name": "muonTrig",
            "collection": "Muon",
            "binning_variables": ("ev, source, nsig: np.abs(ev.Muon.eta)", "ev, source, nsig: ev.Muon_ptShift(ev, source, nsig)"),
            "weighted_paths": [
                (19.7, datapath + "/muons/muon_trigger_IsoMu24_OR_IsoTkMu24_runBCDEF.txt"),
                (16.2, datapath + "/muons/muon_trigger_IsoMu24_OR_IsoTkMu24_runGH.txt"),
            ],
            "add_syst": "ev, source, nsig: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
            "nuisances": ["muonTrig", "muonPtScale"],
        },
    ],
    data = False,
)
weight_taus = Readers.WeightObjects(
    name = "weight_taus",
    correctors = [
        {
            "name": "tauIdTight",
            "collection": "Tau",
            "binning_variables": ("ev, source, nsig: ev.Tau_ptShift(ev, source, nsig)",),
            "weighted_paths": [(1, datapath+"/taus/tau_id_tight.txt")],
            "add_syst": "ev, source, nsig: 0.05*awk.JaggedArray.ones_like(ev.Tau.eta)",
            "nuisances": ["tauIdTight", "tauEnergyScale"],
        },
    ],
    data = False,
)
weight_photon = Readers.WeightObjects(
    name = "weight_photon",
    correctors = [
        {
            "name": "photonIdLoose",
            "collection": "Photon",
            "binning_variables": ("ev, source, nsig: ev.Photon.eta", "ev, source, nsig: ev.Photon_ptShift(ev, source, nsig)"),
            "weighted_paths": [(1, datapath+"/photons/photon_cutbasedid_loose.txt")],
            "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Photon.eta)",
            "nuisances": ["photonIdLoose", "photonEnergyScale"],
        }, {
            "name": "photonPixelSeedVeto",
            "collection": "Photon",
            "binning_variables": ("ev, source, nsig: ev.Photon.r9", "ev, source, nsig: np.abs(ev.Photon.eta)", "ev, source, nsig: ev.Photon_ptShift(ev, source, nsig)"),
            "weighted_paths": [(1, datapath+"/photons/photon_pixelseedveto.txt")],
            "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Photon.eta)",
            "nuisances": ["photonPixelSeedVeto", "photonEnergyScale"],
        },
    ],
    data = False,
)

weight_btags = Readers.WeightBTagging(
    name = "weight_btags",
    operating_point = "medium",
    threshold = 0.8484,
    measurement_types = {"b": "comb", "c": "comb", "udsg": "incl"},
    calibration_file = datapath+"/btagging/CSVv2_Moriond17_B_H_params.csv",
    data = False,
)

weight_qcd_ewk = Readers.WeightQcdEwk(
    name = "weight_qcd_ewk",
    input_paths = {
        "ZJetsToNuNu": (datapath+"/qcd_ewk/vvj.dat", "vvj_pTV_{}"),
        "WJetsToLNu":  (datapath+"/qcd_ewk/evj.dat", "evj_pTV_{}"),
        "DYJetsToLL":  (datapath+"/qcd_ewk/eej.dat", "eej_pTV_{}"),
    },
    underflow = True,
    overflow = True,
    formula = "(K_NLO + d1k_qcd*d1K_NLO + d2k_qcd*d2K_NLO + d3k_qcd*d3K_NLO)",
    params = ["K_NLO", "d1K_NLO", "d2K_NLO", "d3K_NLO"],
    variation_names = ["d1k_qcd", "d2k_qcd", "d3k_qcd"],
    data = False,
)

weight_prefiring = Readers.WeightPreFiring(
    name = "weight_prefiring",
    jet_eff_map_path = datapath+"/prefiring/L1prefiring_jetpt_2016BtoH.txt",
    photon_eff_map_path = datapath+"/prefiring/L1prefiring_photonpt_2016BtoH.txt",
    jet_selection = "ev, source, nsig: (ev.Jet_ptShift(ev, source, nsig)>20) & ((2<np.abs(ev.Jet_eta)) & (np.abs(ev.Jet_eta)<3))",
    photon_selection = "ev, source, nsig: (ev.Photon_ptShift(ev, source, nsig)>20) & ((2<np.abs(ev.Photon_eta)) & (np.abs(ev.Photon_eta)<3))",
    syst = 0.2,
    data = False,
)

selection_producer = Readers.SelectionProducer(
    name = "selection_producer",
)

sqlite_reader = Collectors.SqliteReader(
    name = "sqlite_reader",
    cfg = os.path.join(collpath, "Sqlite_cfg.yaml"),
)

sequence = [
    # Setup caching, nsig and source
    (event_tools, NullCollector()),
    # Creates object collections accessible through the event variable. e.g.
    # event.Jet.pt rather than event.Jet_pt.
    (collection_creator, NullCollector()),
    # Try to keep GenPart branch stuff before everything else. It's quite big
    # and is deleted after use. Don't want to add the memory consumption of
    # this with all other branches
    (gen_boson_producer, NullCollector()),
    (lhe_part_assigner, NullCollector()),
    (gen_part_assigner, NullCollector()),
    (jec_variations, NullCollector()),
    # Cross cleaning must be placed after the veto and selection collections
    # are created. They update the selection flags produced in skim_collections
    # Next 4 modules depend on each other. Turning one off breaks the others.
    (object_functions, NullCollector()),
    (skim_collections, NullCollector()),
    (tau_cross_cleaning, NullCollector()),
    (jet_cross_cleaning, NullCollector()),
    (event_functions, NullCollector()),
    # Readers which create a mask for the event. Doesn't apply it, just stores
    # the mask as an array of booleans
    (trigger_checker, NullCollector()),
    (certified_lumi_checker, NullCollector()),
    (selection_producer, NullCollector()),
    # Weighters. The generally just apply to MC and that logic is dealt with by
    # the ScribblerWrapper.
    (weight_xsection_lumi, NullCollector()),
    (weight_pdf_scale, NullCollector()),
    (weight_pu, NullCollector()),
    (weight_met_trigger, NullCollector()),
    (weight_electrons, NullCollector()),
    (weight_muons, NullCollector()),
    (weight_taus, NullCollector()),
    (weight_photon, NullCollector()),
    (weight_btags, NullCollector()),
    (weight_qcd_ewk, NullCollector()),
    (weight_prefiring, NullCollector()),
    # Add collectors (with accompanying readers) at the end so that all
    # event attributes are available to them
    (sqlite_reader, NullCollector()),
]
