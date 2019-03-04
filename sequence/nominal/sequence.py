from alphatwirl.loop import NullCollector
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
toppath = os.path.abspath(os.environ["TOPDIR"])
datapath = os.path.join(toppath, "data")
collpath = os.path.join(toppath, "sequence", "Collectors")
drawpath = os.path.join(toppath, "drawing")

import Readers
import Collectors

event_tools = Readers.EventTools(
    name = "event_tools",
    maxsize = int(12*1024**3), # 12 GB
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
    data = False,
)
gen_part_assigner = Readers.GenPartAssigner(
    name = "gen_part_assigner",
    data = False,
)

jec_variations = Readers.JecVariations(
    name = "jec_variations",
    jes_unc_file = datapath + "/jecs/Summer16_23Sep2016V4_MC_UncertaintySources_AK4PFchs.txt",
    jer_sf_file = datapath + "/jecs/Spring16_25nsV10a_MC_SF_AK4PFchs.txt",
    jer_file = datapath + "/jecs/Spring16_25nsV10_MC_PtResolution_AK4PFchs.txt",
    apply_jer_corrections = True,
    jes_regex = "jes(?P<source>.*)(Up|Down)",
    unclust_threshold = 15.,
    data = False,
)

object_functions = Readers.ObjectFunctions(
    name = "object_functions",
    selections = [
        ("Jet", "JetVeto", True),
        ("Jet", "JetVetoNoSelection", True),
        ("Jet", "JetSelection", True),
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
)

jet_cross_cleaning = Readers.ObjectCrossCleaning(
    name = "jet_cross_cleaning",
    collections = ("Jet",),
    ref_collections = ("MuonVeto", "ElectronVeto", "PhotonVeto", "TauVeto"),
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
            "binning_variables": ("ev: ev.Electron.eta", "ev: ev.Electron_ptShift(ev)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_idiso_tight.txt")],
            "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Electron.eta)",
        }, {
            "name": "eleIdIsoVeto",
            "collection": "Electron",
            "binning_variables": ("ev: ev.Electron.eta", "ev: ev.Electron_ptShift(ev)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_idiso_veto.txt")],
            "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Electron.eta)",
        }, {
            "name": "eleReco",
            "collection": "Electron",
            "binning_variables": ("ev: ev.Electron.eta", "ev: ev.Electron_ptShift(ev)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_reconstruction.txt")],
            "add_syst": "ev: 0.01*((ev.Electron_ptShift(ev)<20) | (ev.Electron_ptShift(ev)>80))",
        }, {
            "name": "eleTrig",
            "collection": "Electron",
            "binning_variables": ("ev: ev.Electron_ptShift(ev)", "ev: np.abs(ev.Electron.eta)"),
            "weighted_paths": [(1, datapath+"/electrons/electron_trigger_v2.txt")],
            "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Electron.eta)",
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
            "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
            "weighted_paths": [(19.7, datapath+"/muons/muon_id_loose_runBCDEF.txt"),
                               (16.2, datapath+"/muons/muon_id_loose_runGH.txt")],
            "add_syst": "ev: 0.01*awk.JaggedArray.ones_like(ev.Muon.eta)",
        }, {
            "name": "muonIdLoose",
            "collection": "Muon",
            "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
            "weighted_paths": [(19.7, datapath+"/muons/muon_id_loose_runBCDEF.txt"),
                               (16.2, datapath+"/muons/muon_id_loose_runGH.txt")],
            "add_syst": "ev: 0.01*awk.JaggedArray.ones_like(ev.Muon.eta)",
        }, {
            "name": "muonIsoTight",
            "collection": "Muon",
            "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
            "weighted_paths": [(19.7, datapath+"/muons/muon_iso_tight_tightID_runBCDEF.txt"),
                               (16.2, datapath+"/muons/muon_iso_tight_tightID_runGH.txt")],
            "add_syst": "ev: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
        }, {
            "name": "muonIsoLoose",
            "collection": "Muon",
            "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
            "weighted_paths": [(19.7, datapath+"/muons/muon_iso_loose_looseID_runBCDEF.txt"),
                               (16.2, datapath+"/muons/muon_iso_loose_looseID_runGH.txt")],
            "add_syst": "ev: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
        }, {
            "name": "muonTrig",
            "collection": "Muon",
            "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
            "weighted_paths": [(19.7, datapath + "/muons/muon_trigger_IsoMu24_OR_IsoTkMu24_runBCDEF.txt"),
                               (16.2, datapath + "/muons/muon_trigger_IsoMu24_OR_IsoTkMu24_runGH.txt")],
            "add_syst": "ev: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
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
            "binning_variables": ("ev: ev.Tau_ptShift(ev)",),
            "weighted_paths": [(1, datapath+"/taus/tau_id_tight.txt")],
            "add_syst": "ev: 0.05*awk.JaggedArray.ones_like(ev.Tau.eta)",
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
            "binning_variables": ("ev: ev.Photon.eta", "ev: ev.Photon_ptShift(ev)"),
            "weighted_paths": [(1, datapath+"/photons/photon_cutbasedid_loose.txt")],
            "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Photon.eta)",
        }, {
            "name": "photonPixelSeedVeto",
            "collection": "Photon",
            "binning_variables": ("ev: ev.Photon.r9", "ev: np.abs(ev.Photon.eta)", "ev: ev.Photon_ptShift(ev)"),
            "weighted_paths": [(1, datapath+"/photons/photon_pixelseedveto.txt")],
            "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Photon.eta)",
        },
    ],
    data = False,
)

weight_btags = Readers.WeightBTagging(
    name = "weight_btags",
    operating_point = "medium",
    threshold = 0.8484,
    measurement_types = {"b": "comb", "c": "comb", "udsg": "incl"},
    calibration_file = datapath+"/btagging/CSVv2_Moriond17_B_H.csv",
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
    formula = "((K_NNLO + d1k_qcd*d1K_NNLO + d2k_qcd*d2K_NNLO + d3k_qcd*d3K_NNLO)"\
              " /(K_NLO + d1k_qcd*d1K_NLO + d2k_qcd*d2K_NLO + d3k_qcd*d3K_NLO))"\
              "*(1 + kappa_EW + d1k_ew*d1kappa_EW + isz*(d2k_ew_z*d2kappa_EW + d3k_ew_z*d3kappa_EW)"\
                                                 "+ isw*(d2k_ew_w*d2kappa_EW + d3k_ew_w*d3kappa_EW))"\
              "+ dk_mix*dK_NLO_mix",
    params = ["K_NLO", "d1K_NLO", "d2K_NLO", "d3K_NLO", "K_NNLO", "d1K_NNLO",
              "d2K_NNLO", "d3K_NNLO", "kappa_EW", "d1kappa_EW", "d2kappa_EW",
              "d3kappa_EW", "dK_NLO_mix"],
    nuisances = ["d1k_qcd", "d2k_qcd", "d3k_qcd", "d1k_ew", "d2k_ew_z",
                 "d2k_ew_w", "d3k_ew_z", "d3k_ew_w", "dk_mix"],
#    formula = "1 + kappa_EW + d1k_ew*d1kappa_EW + isz*(d2k_ew_z*d2kappa_EW + d3k_ew_z*d3kappa_EW)"\
#                                               "+ isw*(d2k_ew_w*d2kappa_EW + d3k_ew_w*d3kappa_EW)",
#    params = ["kappa_EW", "d1kappa_EW", "d2kappa_EW", "d3kappa_EW"],
#    nuisances = ["d1k_ew", "d2k_ew_z", "d2k_ew_w", "d3k_ew_z", "d3k_ew_w"],
    data = False,
)

weight_prefiring = Readers.WeightPreFiring(
    name = "weight_prefiring",
    jet_eff_map_path = datapath+"/prefiring/L1prefiring_jetpt_2016BtoH.txt",
    photon_eff_map_path = datapath+"/prefiring/L1prefiring_photonpt_2016BtoH.txt",
    jet_selection = "ev: (ev.Jet_ptShift(ev)>20) & ((2<np.abs(ev.Jet_eta)) & (np.abs(ev.Jet_eta)<3))",
    photon_selection = "ev: (ev.Photon_ptShift(ev)>20) & ((2<np.abs(ev.Photon_eta)) & (np.abs(ev.Photon_eta)<3))",
    syst = 0.2,
    data = False,
)

selection_producer = Readers.SelectionProducer(
    name = "selection_producer",
)

weight_producer = Readers.WeightProducer(
    name = "weight_producer",
)

hist_reader = Collectors.HistReader(
    name = "hist_reader",
    cfg = os.path.join(collpath, "Histogrammer_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
    split_samples = {
        "DYJetsToLL": {
            "DYJetsToEE": ["ev: ev.LeptonIsElectron"],
            "DYJetsToMuMu": ["ev: ev.LeptonIsMuon"],
            "DYJetsToTauLTauL": ["ev: ev.LeptonIsTau & (ev.nGenTauL==2)"],
            "DYJetsToTauLTauH": ["ev: ev.LeptonIsTau & (ev.nGenTauL==1)"],
            "DYJetsToTauHTauH": ["ev: ev.LeptonIsTau & (ev.nGenTauL==0)"],
        },
        "WJetsToLNu": {
            "WJetsToENu": ["ev: ev.LeptonIsElectron"],
            "WJetsToMuNu": ["ev: ev.LeptonIsMuon"],
            "WJetsToTauLNu": ["ev: ev.LeptonIsTau & (ev.nGenTauL==1)"],
            "WJetsToTauHNu": ["ev: ev.LeptonIsTau & (ev.nGenTauL==0)"],
        },
    },
)
hist_collector = Collectors.HistCollector(
    name = "hist_collector",
    plot = True,
    cfg = os.path.join(collpath, "Histogrammer_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

hist2d_reader = Collectors.Hist2DReader(
    name = "hist2d_reader",
    cfg = os.path.join(collpath, "Histogrammer2D_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)
hist2d_collector = Collectors.Hist2DCollector(
    name = "hist2d_collector",
    plot = True,
    cfg = os.path.join(collpath, "Histogrammer2D_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

gen_stitching_reader = Collectors.GenStitchingReader(
    name = "gen_stitching_reader",
    cfg = os.path.join(collpath, "GenStitching_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)
gen_stitching_collector = Collectors.GenStitchingCollector(
    name = "gen_stitching_collector",
    plot = True,
    cfg = os.path.join(collpath, "GenStitching_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

met_response_resolution_reader = Collectors.MetResponseResolutionReader(
    name = "met_response_resolution_reader",
    cfg = os.path.join(collpath, "MetResponseResolution_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)
met_response_resolution_collector = Collectors.MetResponseResolutionCollector(
    name = "met_response_resolution_collector",
    plot = True,
    cfg = os.path.join(collpath, "MetResponseResolution_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

qcd_ewk_corrections_reader = Collectors.QcdEwkCorrectionsReader(
    name = "qcd_ewk_corrections_reader",
    cfg = os.path.join(collpath, "QcdEwkCorrections_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)
qcd_ewk_corrections_collector = Collectors.QcdEwkCorrectionsCollector(
    name = "qcd_ewk_corrections_collector",
    plot = True,
    cfg = os.path.join(collpath, "QcdEwkCorrections_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

systematics_reader = Collectors.SystematicsReader(
    name = "systematics_reader",
    cfg = os.path.join(collpath, "Systematics_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)
systematics_collector = Collectors.SystematicsCollector(
    name = "systematics_collector",
    plot = True,
    cfg = os.path.join(collpath, "Systematics_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

trigger_efficiency_reader = Collectors.TriggerEfficiencyReader(
    name = "trigger_efficiency_reader",
    cfg = os.path.join(collpath, "TriggerEfficiency_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)
trigger_efficiency_collector = Collectors.TriggerEfficiencyCollector(
    name = "trigger_efficiency_collector",
    plot = True,
    cfg = os.path.join(collpath, "TriggerEfficiency_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

qcd_estimation_reader = Collectors.QcdEstimationReader(
    name = "qcd_estimation_reader",
    cfg = os.path.join(collpath, "QcdEstimation_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)
qcd_estimation_collector = Collectors.QcdEstimationCollector(
    name = "qcd_estimation_collector",
    plot = True,
    cfg = os.path.join(collpath, "QcdEstimation_cfg.yaml"),
    drawing_cfg = os.path.join(drawpath, "config.yaml"),
)

sequence = [
    # Setup caching, nsig and source
    (event_tools, NullCollector()),
    # Creates object collections accessible through the event variable. e.g.
    # event.Jet.pt rather than event.Jet_pt.
    (collection_creator, NullCollector()),
    # selection and weight producers. They only create functions and hence can
    # be placed near the start
    (selection_producer, NullCollector()),
    (weight_producer, NullCollector()),
    # # Try to keep GenPart branch stuff before everything else. It's quite big
    # # and is deleted after use. Don't want to add the memory consumption of
    # # this with all other branches
    (gen_boson_producer, NullCollector()),
    (lhe_part_assigner, NullCollector()),
    (gen_part_assigner, NullCollector()),
    (jec_variations, NullCollector()),
    (object_functions, NullCollector()),
    (event_functions, NullCollector()),
    # # Cross cleaning must be placed after the veto and selection collections
    # # are created. They update the selection flags produced in skim_collections
    (skim_collections, NullCollector()),
    (tau_cross_cleaning, NullCollector()),
    (jet_cross_cleaning, NullCollector()),
    # # Readers which create a mask for the event. Doesn't apply it, just stores
    # # the mask as an array of booleans
    (trigger_checker, NullCollector()),
    (certified_lumi_checker, NullCollector()),
    # # Weighters. The generally just apply to MC and that logic is dealt with by
    # # the ScribblerWrapper.
    (weight_xsection_lumi, NullCollector()),
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
    (hist_reader, hist_collector),
    #(hist2d_reader, hist2d_collector),
    #(gen_stitching_reader, gen_stitching_collector),
    #(met_response_resolution_reader, met_response_resolution_collector),
    #(qcd_ewk_corrections_reader, qcd_ewk_corrections_collector),
    #(systematics_reader, systematics_collector),
    #(trigger_efficiency_reader, trigger_efficiency_collector),
    #(qcd_estimation_reader, qcd_estimation_collector),
]
