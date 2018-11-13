import numpy as np
import matplotlib.pyplot as plt
from utils.Colours import colours_dict

inf = np.infty
pi = np.pi+0.00001

# dataset-cutflows split into regions
monojet_categories = [("MET", "None"),
                      ("MET", "Monojet"), ("MET", "MonojetSB"), ("MET", "MonojetSR"),
                      ("MET", "MonojetQCD"), ("MET", "MonojetQCDSB"), ("MET", "MonojetQCDSR")]

muon_met_categories = [("MET", "SingleMuon"), ("MET", "SingleMuonSB"), ("MET", "SingleMuonSR")]
muon_mu_categories = [("SingleMuon", "SingleMuon"), ("SingleMuon", "SingleMuonSB"), ("SingleMuon", "SingleMuonSR")]
dimuon_met_categories = [("MET", "DoubleMuon"), ("MET", "DoubleMuonSB"), ("MET", "DoubleMuonSR")]
dimuon_mu_categories = [("SingleMuon", "DoubleMuon"),("SingleMuon", "DoubleMuonSB"), ("SingleMuon", "DoubleMuonSR")]
ele_categories = [("SingleElectron", "SingleElectron"), ("SingleElectron", "SingleElectronSB"), ("SingleElectron", "SingleElectronSR")]
diele_categories = [("SingleElectron", "DoubleElectron"), ("SingleElectron", "DoubleElectronSB"), ("SingleElectron", "DoubleElectronSR")]

categories = monojet_categories\
        + muon_met_categories + muon_mu_categories\
        + dimuon_met_categories + dimuon_mu_categories\
        + ele_categories + diele_categories

monojet_variations = [
    ("nominal",       "ev: ev.Weight_{dataset}"),
    ("pileupUp",      "ev: ev.Weight_{dataset}*ev.Weight_pileupUp"),
    ("pileupDown",    "ev: ev.Weight_{dataset}*ev.Weight_pileupDown"),
    ("metTrigSFUp",   "ev: ev.Weight_{dataset}*ev.Weight_metTrigSFUp"),
    ("metTrigSFDown", "ev: ev.Weight_{dataset}*ev.Weight_metTrigSFDown"),
    ("d1k_ewUp",      "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d1k_ewUp"),
    ("d1k_ewDown",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d1k_ewDown"),
    ("d2k_ew_zUp",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_zUp"),
    ("d2k_ew_zDown",  "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_zDown"),
    ("d2k_ew_wUp",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_wUp"),
    ("d2k_ew_wDown",  "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_wDown"),
    ("d3k_ew_zUp",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_zUp"),
    ("d3k_ew_zDown",  "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_zDown"),
    ("d3k_ew_wUp",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_wUp"),
    ("d3k_ew_wDown",  "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_wDown"),
]

muon_met_variations = monojet_variations + [
    ("muonIdUp",      "ev: ev.Weight_{dataset}*ev.Weight_muonIdUp"),
    ("muonIdDown",    "ev: ev.Weight_{dataset}*ev.Weight_muonIdDown"),
    ("muonIsoUp",     "ev: ev.Weight_{dataset}*ev.Weight_muonIsoUp"),
    ("muonIsoDown",   "ev: ev.Weight_{dataset}*ev.Weight_muonIsoDown"),
    ("muonTrackUp",   "ev: ev.Weight_{dataset}*ev.Weight_muonTrackUp"),
    ("muonTrackDown", "ev: ev.Weight_{dataset}*ev.Weight_muonTrackDown"),
    ("muonTrigUp",    "ev: ev.Weight_{dataset}*ev.Weight_muonTrigUp"),
    ("muonTrigDown",  "ev: ev.Weight_{dataset}*ev.Weight_muonTrigDown"),
]

muon_mu_variations = muon_met_variations + [
    ("muonTrigUp",    "ev: ev.Weight_{dataset}*ev.Weight_muonTrigUp"),
    ("muonTrigDown",  "ev: ev.Weight_{dataset}*ev.Weight_muonTrigDown"),
]

ele_variations = monojet_variations + [
    ("eleIdIsoUp", "ev: ev.Weight_{dataset}*ev.Weight_eleIdIsoUp"),
    ("eleIdIsoDown", "ev: ev.Weight_{dataset}*ev.Weight_eleIdIsoDown"),
    ("eleRecoUp", "ev: ev.Weight_{dataset}*ev.Weight_eleRecoUp"),
    ("eleRecoDown", "ev: ev.Weight_{dataset}*ev.Weight_eleRecoDown"),
    ("eleTrigUp", "ev: ev.Weight_{dataset}*ev.Weight_eleTrigUp"),
    ("eleTrigDown", "ev: ev.Weight_{dataset}*ev.Weight_eleTrigDown"),
]

jes_variation_names = [
    "Total",
    #"AbsoluteStat", "AbsoluteScale", "AbsoluteMPFBias", "Fragmentation", "SinglePionECAL", "SinglePionHCAL",
    #"FlavorQCD", "TimePtEta", "RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF", "RelativePtBB",
    #"RelativePtEC1", "RelativePtEC2", "RelativePtHF", "RelativeBal", "RelativeFSR", "RelativeStatFSR", "RelativeStatEC",
    #"RelativeStatHF", "PileUpDataMC", "PileUpPtRef", "PileUpPtBB", "PileUpPtEC1", "PileUpPtEC2", "PileUpPtHF",
]
jes_variations = [
    "jes"+var+"Up" for var in jes_variation_names
] + [
    "jes"+var+"Down" for var in jes_variation_names
] + [
    "jerUp", "jerDown",
    "unclustUp", "unclustDown",
]

histogrammer_cfgs = [
    {
        "name": "METnoX_pt",
        "categories": monojet_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": monojet_variations,
    }, {
        "name": "METnoX_pt",
        "categories": muon_met_categories + dimuon_met_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": muon_met_variations,
    }, {
        "name": "METnoX_pt",
        "categories": muon_mu_categories + dimuon_mu_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": muon_mu_variations,
    }, {
        "name": "METnoX_pt",
        "categories": ele_categories + diele_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": ele_variations,
    }
] + [
    {
        "name": "METnoX_pt",
        "categories": [(d, c+variation) for d, c in categories],
        "variables": ["ev: ev.METnoX_pt{}".format(variation)],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": [(variation, "ev: ev.Weight_{dataset}")],
    } for variation in jes_variations
]

cmap = plt.cm.viridis
colors = [cmap(i) for i in np.linspace(0, 1, 18)]
sample_colours = {
    "nominal":   "black",
    "mcstat":    colors[0],
    "pileup":    colors[1],
    "metTrigSF": colors[2],
    "muonId":    colors[3],
    "muonIso":   colors[4],
    "muonTrack": colors[5],
    "muonTrig":  colors[6],
    "jesTotal":  colors[7],
    "jer":       colors[8],
    "unclust":   colors[9],
    "d1k_ew":    colors[10],
    "d2k_ew_z":  colors[11],
    "d2k_ew_w":  colors[12],
    "d3k_ew_z":  colors[13],
    "d3k_ew_w":  colors[14],
    "eleIdIso":  colors[15],
    "eleReco":   colors[16],
    "eleTrig":   colors[17],
}

sample_names = {
    "nominal":            r'Nominal',
    "mcstat":             r'MC stat',
    "pileup":             r'Pileup',
    "metTrigSF":          r'$E_{T}^{miss}$ Trig',
    "muonId":             r'Muon ID',
    "muonIso":            r'Muon Iso',
    "muonTrack":          r'Muon Track',
    "muonTrig":           r'Muon Trig',
    "jer":                r'JER',
    "jesTotal":           r'JES',
    "jesAbsoluteStat":    r'JES abs stat',
    "jesAbsoluteScale":   r'JES abs scale',
    "jesAbsoluteMPFBias": r'JES abs ISR+FSR bias',
    "jesFragmentation":   r'JES frag',
    "jesSinglePionECAL":  r'JES single $\pi$ ECAL',
    "jesSinglePionHCAL":  r'JES single $\pi$ HCAL',
    "jesFlavorQCD":       r'JES flavour QCD',
    "jesTimePtEta":       r'JES time $p_{\rm{T}}$--$\eta$',
    "jesRelativeJEREC1":  r'JER EC1',
    "jesRelativeJEREC2":  r'JER EC2',
    "jesRelativeJERHF":   r'JER HF',
    "jesRelativePtBB":    r'JES $p_{\rm{T}}$ BB',
    "jesRelativePtEC1":   r'JES $p_{\rm{T}}$ EC1',
    "jesRelativePtEC2":   r'JES $p_{\rm{T}}$ EC2',
    "jesRelativePtHF":    r'JES $p_{\rm{T}}$ HF',
    "jesRelativeBal":     r'JES Bal',
    "jesRelativeFSR":     r'JES ISR+FSR',
    "jesRelativeStatFSR": r'JES ISR+FSR stat',
    "jesRelativeStatEC":  r'JES EC stat',
    "jesRelativeStatHF":  r'JES HF stat',
    "jesPileUpDataMC":    r'JES PU Data/MC',
    "jesPileUpPtRef":     r'JES PU $p_{\rm{T}}$ ref',
    "jesPileUpPtBB":      r'JES PU $p_{\rm{T}}$ BB',
    "jesPileUpPtEC1":     r'JES PU $p_{\rm{T}}$ EC1',
    "jesPileUpPtEC2":     r'JES PU $p_{\rm{T}}$ EC2',
    "jesPileUpPtHF":      r'JES PU $p_{\rm{T}}$ HF',
    "unclust":            r'Unclust En',
    "d1k_ew":             r'$\delta^{(1)}\kappa_{EW}$',
    "d2k_ew_z":           r'$\delta^{(2)}\kappa_{EW}^{Z}$',
    "d2k_ew_w":           r'$\delta^{(2)}\kappa_{EW}^{W}$',
    "d3k_ew_z":           r'$\delta^{(3)}\kappa_{EW}^{Z}$',
    "d3k_ew_w":           r'$\delta^{(3)}\kappa_{EW}^{W}$',
    "eleIdIso":           r'Ele ID/Iso',
    "eleReco":            r'Ele Reco',
    "eleTrig":            r'Ele Trig',

    "znunu":    r'$Z_{\nu\nu}$+jets',
    "wlnu":     r'$W_{l\nu}$+jets',
    "bkg":      r'Bkg',
    "qcd":      r'QCD',
    "zmumu":    r'$Z/\gamma^{*}_{\mu\mu}$+jets',
    "zee":      r'$Z/\gamma^{*}_{e e}$+jets',
}

axis_label = {
    "METnoX_pt": r'$E_{T}^{miss}$ (GeV)',
}
