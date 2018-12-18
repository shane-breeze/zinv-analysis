import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import Readers
import Collectors
from event_selection import event_selection
from physics_object_selection import physics_object_selection
from alphatwirl.loop import NullCollector

import os
datapath = os.path.join(os.environ["TOPDIR"], "data")

jes_variations = [
    "Total",
    #"AbsoluteStat", "AbsoluteScale", "AbsoluteMPFBias", "Fragmentation", "SinglePionECAL", "SinglePionHCAL",
    #"FlavorQCD", "TimePtEta", "RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF", "RelativePtBB",
    #"RelativePtEC1", "RelativePtEC2", "RelativePtHF", "RelativeBal", "RelativeFSR", "RelativeStatFSR", "RelativeStatEC",
    #"RelativeStatHF", "PileUpDataMC", "PileUpPtRef", "PileUpPtBB", "PileUpPtEC1", "PileUpPtEC2", "PileUpPtHF",
]
variations_noupdown = ["jes"+j for j in jes_variations] + ["jer", "unclust"]

all_variations = [var+"Up" for var in variations_noupdown]\
        + [var+"Down" for var in variations_noupdown]

certified_lumi_checker = Readers.CertifiedLumiChecker(
    name = "certified_lumi_checker",
    lumi_json_path = datapath + "/json/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt",
    mc = False,
)

trigger_checker = Readers.TriggerChecker(
    name = "trigger_checker",
)

collection_creator = Readers.CollectionCreator(
    name = "collection_creator",
    collections = ["CaloMET", "MET", "Jet", "Electron", "Muon", "Photon", "Tau",
                   "GenMET", "GenPart", "GenJet", "GenDressedLepton", "LHEPart"],
)

skim_collections = Readers.SkimCollections(
    name = "skim_collections",
    selection_dict = physics_object_selection.selection_dict,
    variations = all_variations,
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
    variations = all_variations,
)

jec_variations = Readers.JecVariations(
    name = "jec_variations",
    jes_unc_file = datapath + "/jecs/Summer16_23Sep2016V4_MC_UncertaintySources_AK4PFchs.txt",
    jer_sf_file = datapath + "/jecs/Spring16_25nsV10a_MC_SF_AK4PFchs.txt",
    jer_file = datapath + "/jecs/Spring16_25nsV10_MC_PtResolution_AK4PFchs.txt",
    apply_jer_corrections = True,
    do_jes = True,
    do_jer = True,
    do_unclust = True,
    sources = jes_variations,
)

event_sums_producer = Readers.EventSumsProducer(
    name = "event_sums_producer",
    variations = all_variations,
)
signal_region_blinder = Readers.SignalRegionBlinder(
    name = "signal_region_blinder",
    blind = True,
    apply_to_mc = False,
)
inv_mass_producer = Readers.InvMassProducer(
    name = "inv_mass_producer",
)
gen_boson_producer = Readers.GenBosonProducer(
    name = "gen_boson_producer",
    data = False,
)
lhe_part_assigner = Readers.LHEPartAssigner(
    name = "lhe_part_assigner",
    data = False,
)

weight_creator = Readers.WeightCreator(
    name = "weight_creator",
)
weight_xsection_lumi = Readers.WeightXsLumi(
    name = "weight_xsection_lumi",
    data = False,
)
weight_pu = Readers.WeightPileup(
    name = "weight_pu",
    correction_file = datapath + "/pileup/nTrueInt_corrections.txt",
    overflow = True,
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
weight_muons = Readers.WeightObjects(
    name = "weight_muons",
    dataset_applicators = {
        "MET": "muonId*muonIso*muonTrack",
        "SingleMuon": "muonId*muonIso*muonTrack*muonTrig",
        "SingleElectron": "muonId*muonIso*muonTrack",
    },
    correctors = [
        {
            "name": "muonId",
            "collection": "MuonSelection",
            "binning_variables": ("mu: mu.pt.content", "mu: np.abs(mu.eta.content)"),
            "weighted_paths": [(19.7, datapath+"/muons/muon_id_runBCDEF.txt"),
                               (16.2, datapath+"/muons/muon_id_runGH.txt")],
            "add_syst": 0.01,
        }, {
            "name": "muonIso",
            "collection": "MuonSelection",
            "binning_variables": ("mu: mu.pt.content", "mu: np.abs(mu.eta.content)"),
            "weighted_paths": [(19.7, datapath+"/muons/muon_isolation_runBCDEF.txt"),
                               (16.2, datapath+"/muons/muon_isolation_runGH.txt")],
            "add_syst": 0.005,
        }, {
            "name": "muonTrack",
            "collection": "MuonSelection",
            "binning_variables": ("mu: mu.eta.content", ),
            "weighted_paths": [(1., datapath+"/muons/muon_tracking.txt")],
        }, {
            "name": "muonTrig",
            "collection": "MuonSelection",
            "binning_variables": ("mu: mu.pt.content", "mu: np.abs(mu.eta.content)"),
            "weighted_paths": [(19.7, datapath + "/muons/muon_trigger_runBCDEF.txt"),
                               (16.2, datapath + "/muons/muon_trigger_runGH.txt")],
            "any_pass": True,
            "add_syst": 0.005,
        },
    ],
    data = False,
)
weight_electrons = Readers.WeightObjects(
    name = "weight_electrons",
    dataset_applicators = {
        "MET": "eleIdIso*eleReco",
        "SingleMuon": "eleIdIso*eleReco",
        "SingleElectron": "eleIdIso*eleReco*eleTrig",
    },
    correctors = [
        {
            "name": "eleIdIso",
            "collection": "ElectronSelection",
            "binning_variables": ("e: e.eta.content", "e: e.pt.content"),
            "weighted_paths": [(1, datapath+"/electrons/electron_idiso.txt")],
        }, {
            "name": "eleReco",
            "collection": "ElectronSelection",
            "binning_variables": ("e: e.eta.content", "e: e.pt.content"),
            "weighted_paths": [(1, datapath+"/electrons/electron_reconstruction.txt")],
        }, {
            "name": "eleTrig",
            "collection": "ElectronSelection",
            "binning_variables": ("e: e.pt.content", "e: e.eta.content"),
            "weighted_paths": [(1, datapath+"/electrons/electron_trigger.txt")],
            "any_pass": True,
        },
    ],
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
)
#weight_qcd_ewk = Readers.WeightQcdEwk(
#    name = "weight_qcd_ewk",
#    input_paths = {
#        "ZJetsToNuNu": (datapath+"/qcd_ewk/vvj.dat", "vvj_pTV_{}"),
#        "WJetsToLNu":  (datapath+"/qcd_ewk/evj.dat", "evj_pTV_{}"),
#        "DYJetsToLL":  (datapath+"/qcd_ewk/eej.dat", "eej_pTV_{}"),
#    },
#    underflow = True,
#    overflow = True,
#    formula = "1 + kappa_EW + d1k_ew*d1kappa_EW + isz*(d2k_ew_z*d2kappa_EW + d3k_ew_z*d3kappa_EW)"\
#                                               "+ isw*(d2k_ew_w*d2kappa_EW + d3k_ew_w*d3kappa_EW)",
#    params = ["kappa_EW", "d1kappa_EW", "d2kappa_EW", "d3kappa_EW"],
#    nuisances = ["d1k_ew", "d2k_ew_z", "d2k_ew_w", "d3k_ew_z", "d3k_ew_w"],
#)

weight_prefiring = Readers.WeightPreFiring(
    name = "weight_prefiring",
    jet_eff_map_path = datapath+"/prefiring/L1prefiring_jetpt_2016BtoH.txt",
    photon_eff_map_path = datapath+"/prefiring/L1prefiring_photonpt_2016BtoH.txt",
    jet_selection = "j: (j.pt>20) & ((2<np.abs(j.eta)) & (np.abs(j.eta)<3))",
    photon_selection = "y: (y.pt>20) & ((2<np.abs(y.eta)) & (np.abs(y.eta)<3))",
    syst = 0.2,
    apply = False,
)

selection_producer = Readers.SelectionProducer(
    name = "selection_producer",
    event_selection = event_selection,
    variations = all_variations,
)

hist_reader = Collectors.HistReader(
    name = "hist_reader",
    cfg = Collectors.Histogrammer_cfg,
)
hist_collector = Collectors.HistCollector(
    name = "hist_collector",
    plot = True,
    cfg = Collectors.Histogrammer_cfg,
)

hist2d_reader = Collectors.Hist2DReader(
    name = "hist2d_reader",
    cfg = Collectors.Histogrammer2D_cfg,
)
hist2d_collector = Collectors.Hist2DCollector(
    name = "hist2d_collector",
    plot = True,
    cfg = Collectors.Histogrammer2D_cfg,
)

gen_stitching_reader = Collectors.GenStitchingReader(
    name = "gen_stitching_reader",
    cfg = Collectors.GenStitching_cfg,
)
gen_stitching_collector = Collectors.GenStitchingCollector(
    name = "gen_stitching_collector",
    plot = True,
    cfg = Collectors.GenStitching_cfg,
)

met_response_resolution_reader = Collectors.MetResponseResolutionReader(
    name = "met_response_resolution_reader",
    cfg = Collectors.MetResponseResolution_cfg,
)
met_response_resolution_collector = Collectors.MetResponseResolutionCollector(
    name = "met_response_resolution_collector",
    plot = True,
    cfg = Collectors.MetResponseResolution_cfg,
    variations = all_variations,
)

qcd_ewk_corrections_reader = Collectors.QcdEwkCorrectionsReader(
    name = "qcd_ewk_corrections_reader",
    cfg = Collectors.QcdEwkCorrections_cfg,
)
qcd_ewk_corrections_collector = Collectors.QcdEwkCorrectionsCollector(
    name = "qcd_ewk_corrections_collector",
    plot = True,
    cfg = Collectors.QcdEwkCorrections_cfg,
)

systematics_reader = Collectors.SystematicsReader(
    name = "systematics_reader",
    cfg = Collectors.Systematics_cfg,
)
systematics_collector = Collectors.SystematicsCollector(
    name = "systematics_collector",
    plot = True,
    cfg = Collectors.Systematics_cfg,
)

trigger_efficiency_reader = Collectors.TriggerEfficiencyReader(
    name = "trigger_efficiency_reader",
    cfg = Collectors.TriggerEfficiency_cfg,
)
trigger_efficiency_collector = Collectors.TriggerEfficiencyCollector(
    name = "trigger_efficiency_collector",
    plot = True,
    cfg = Collectors.TriggerEfficiency_cfg,
)

qcd_estimation_reader = Collectors.QcdEstimationReader(
    name = "qcd_estimation_reader",
    cfg = Collectors.QcdEstimation_cfg,
)
qcd_estimation_collector = Collectors.QcdEstimationCollector(
    name = "qcd_estimation_collector",
    plot = True,
    cfg = Collectors.QcdEstimation_cfg,
)

sequence = [
    # Creates object collections accessible through the event variable. e.g.
    # event.Jet.pt rather than event.Jet_pt. Simpler to pass a collection to
    # functions and allows subcollections (done by skim_collections)
    (collection_creator, NullCollector()),
    # Try to keep GenPart branch stuff before everything else. It's quite big
    # and is deleted after use. Don't want to add the memory consumption of
    # this with all other branches
    (gen_boson_producer, NullCollector()),
    (lhe_part_assigner, NullCollector()),
    (jec_variations, NullCollector()),
    (skim_collections, NullCollector()),
    # Cross cleaning must be placed after the veto and selection collections
    # are created but before they're used anywhere to allow the collection
    # selection mask to be updated
    (tau_cross_cleaning, NullCollector()),
    (jet_cross_cleaning, NullCollector()),
    # General event variable producers
    (event_sums_producer, NullCollector()),
    (inv_mass_producer, NullCollector()),
    # Readers which create a mask for the event. Doesn't apply it, just stores
    # the mask as an array of booleans
    (trigger_checker, NullCollector()),
    (certified_lumi_checker, NullCollector()),
    (signal_region_blinder, NullCollector()),
    # Weighters. Need to add a weight (of ones) to the event first -
    # weight_creator. The generally just apply to MC and that logic it dealt
    # with by the ScribblerWrapper.
    (weight_creator, NullCollector()),
    (weight_xsection_lumi, NullCollector()),
    (weight_pu, NullCollector()),
    (weight_met_trigger, NullCollector()),
    (weight_muons, NullCollector()),
    (weight_electrons, NullCollector()),
    (weight_qcd_ewk, NullCollector()),
    (weight_prefiring, NullCollector()),
    (selection_producer, NullCollector()),
    # Add collectors (with accompanying readers) at the end so that all
    # event attributes are available to them
    (hist_reader, hist_collector),
    (hist2d_reader, hist2d_collector),
    (gen_stitching_reader, gen_stitching_collector),
    (met_response_resolution_reader, met_response_resolution_collector),
    (qcd_ewk_corrections_reader, qcd_ewk_corrections_collector),
    (systematics_reader, systematics_collector),
    (trigger_efficiency_reader, trigger_efficiency_collector),
    (qcd_estimation_reader, qcd_estimation_collector),
]
