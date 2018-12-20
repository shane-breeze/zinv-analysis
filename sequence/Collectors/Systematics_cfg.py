import numpy as np
import matplotlib.pyplot as plt
from utils.Colours import colours_dict

inf = np.infty
pi = np.pi+0.00001

# dataset-cutflows split into regions
monojet_categories = [("MET", "None"),
                      ("MET", "Monojet"), ("MET", "MonojetQCD")]
muon_met_categories = [("MET", "SingleMuon"), ("MET", "SingleMuonPlus"), ("MET", "SingleMuonMinus"),
                       ("MET", "SingleMuonQCD")]
muon_mu_categories = [("SingleMuon", "SingleMuon"), ("SingleMuon", "SingleMuonPlus"), ("SingleMuon", "SingleMuonMinus"),
                      ("SingleMuon", "SingleMuonQCD")]
dimuon_met_categories = [("MET", "DoubleMuon")]
dimuon_mu_categories = [("SingleMuon", "DoubleMuon")]
ele_categories = [("SingleElectron", "SingleElectron"), ("SingleElectron", "SingleElectronPlus"), ("SingleElectron", "SingleElectronMinus"),
                  ("SingleElectron", "SingleElectronQCD")]
diele_categories = [("SingleElectron", "DoubleElectron")]
tau_categories = [("MET", "SingleTau")]

categories = monojet_categories\
        + muon_met_categories + muon_mu_categories\
        + dimuon_met_categories + dimuon_mu_categories\
        + ele_categories + diele_categories\
        + tau_categories

monojet_variations = [
    ("nominal",         "ev: ev.Weight_{dataset}"),
    ("pileupUp",        "ev: ev.Weight_{dataset}*ev.Weight_pileupUp"),
    ("pileupDown",      "ev: ev.Weight_{dataset}*ev.Weight_pileupDown"),
    ("metTrigStatUp",   "ev: ev.Weight_{dataset}*ev.Weight_metTrigStatUp"),
    ("metTrigStatDown", "ev: ev.Weight_{dataset}*ev.Weight_metTrigStatDown"),
    ("metTrigSystUp",   "ev: ev.Weight_{dataset}*ev.Weight_metTrigSystUp"),
    ("metTrigSystDown", "ev: ev.Weight_{dataset}*ev.Weight_metTrigSystDown"),
    ("d1k_qcdUp",       "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d1k_qcdUp"),
    ("d1k_qcdDown",     "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d1k_qcdDown"),
    ("d2k_qcdUp",       "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_qcdUp"),
    ("d2k_qcdDown",     "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_qcdDown"),
    ("d3k_qcdUp",       "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_qcdUp"),
    ("d3k_qcdDown",     "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_qcdDown"),
    ("d1k_ewUp",        "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d1k_ewUp"),
    ("d1k_ewDown",      "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d1k_ewDown"),
    ("d2k_ew_zUp",      "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_zUp"),
    ("d2k_ew_zDown",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_zDown"),
    ("d2k_ew_wUp",      "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_wUp"),
    ("d2k_ew_wDown",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d2k_ew_wDown"),
    ("d3k_ew_zUp",      "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_zUp"),
    ("d3k_ew_zDown",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_zDown"),
    ("d3k_ew_wUp",      "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_wUp"),
    ("d3k_ew_wDown",    "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_d3k_ew_wDown"),
    ("dk_mixUp",        "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_dk_mixUp"),
    ("dk_mixDown",      "ev: ev.Weight_{dataset}*ev.WeightQcdEwk_dk_mixDown"),
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

tau_variations = monojet_variations

jes_variation_names = [
    "jesTotal",
    "jesAbsoluteStat", "jesAbsoluteScale", "jesAbsoluteMPFBias", "jesFragmentation", "jesSinglePionECAL", "jesSinglePionHCAL",
    "jesFlavorQCD", "jesTimePtEta", "jesRelativeJEREC1", "jesRelativeJEREC2", "jesRelativeJERHF", "jesRelativeBal",
    "jesRelativePtBB", "jesRelativePtEC1", "jesRelativePtEC2", "jesRelativePtHF", "jesRelativeStatFSR", "jesRelativeStatEC", "jesRelativeStatHF",
    "jesRelativeFSR", "jesPileUpDataMC", "jesPileUpPtRef", "jesPileUpPtBB", "jesPileUpPtEC1", "jesPileUpPtEC2", "jesPileUpPtHF",
    "jer", "unclust",
]
jes_variations = [var+"Up" for var in jes_variation_names]\
        + [var+"Down" for var in jes_variation_names]

pdf_variations = [
#    ("lhePdf{}".format(i), "ev: ev.Weight_{dataset}"+"*ev.LHEPdfWeightList[:,{0}] if {0} < ev.nLHEPdfWeight[0] and ev.config.dataset.parent not in [\"SingleTop\"] else np.full(ev.size, np.nan)".format(i))
#    for i in range(0,110)
]
scale_variations = [
#    ("lheScale{}".format(i), "ev: ev.Weight_{dataset}"+"*ev.LHEScaleWeightList[:,{0}] if {0} < ev.nLHEScaleWeight[0] and ev.config.dataset.parent not in [\"SingleTop\"] else np.full(ev.size, np.nan)".format(i))
#    for i in (0,1,3,5,7,8)
]

histogrammer_cfgs = [
    {
        "name": "METnoX_pt",
        "categories": monojet_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": monojet_variations+pdf_variations+scale_variations,
    }, {
        "name": "METnoX_pt",
        "categories": muon_met_categories + dimuon_met_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": muon_met_variations+pdf_variations+scale_variations,
    }, {
        "name": "METnoX_pt",
        "categories": muon_mu_categories + dimuon_mu_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": muon_mu_variations+pdf_variations+scale_variations,
    }, {
        "name": "METnoX_pt",
        "categories": ele_categories + diele_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": ele_variations+pdf_variations+scale_variations,
    }, {
        "name": "METnoX_pt",
        "categories": tau_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 201))+[inf]],
        "weights": tau_variations+pdf_variations+scale_variations,
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
    "nominal":     "black",
    "mcstat":      colors[0],
    "pileup":      colors[1],
    "metTrigStat": colors[2],
    "metTrigSyst": colors[2],
    "muonId":      colors[3],
    "muonIso":     colors[4],
    "muonTrack":   colors[5],
    "muonTrig":    colors[6],
    "jesTotal":    colors[7],
    "jer":         colors[8],
    "unclust":     colors[9],
    "d1k_qcd":     colors[10],
    "d2k_qcd":     colors[10],
    "d3k_qcd":     colors[10],
    "d1k_ew":      colors[10],
    "d2k_ew_z":    colors[11],
    "d2k_ew_w":    colors[12],
    "d3k_ew_z":    colors[13],
    "d3k_ew_w":    colors[14],
    "dk_mix":      colors[14],
    "eleIdIso":    colors[15],
    "eleReco":     colors[16],
    "eleTrig":     colors[17],
}
sample_colours.update({"lhePdf{}".format(i): 'blue' for i in range(102)})
sample_colours.update({"lheScale{}".format(i): 'green' for i in range(9)})

sample_names = {
    "nominal":            r'Nominal',
    "mcstat":             r'MC stat',
    "pileup":             r'Pileup',
    "metTrigStat":        r'$p_{\mathrm{T}}^{\mathrm{miss}}$ Trig. stat.',
    "metTrigSyst":        r'$p_{\mathrm{T}}^{\mathrm{miss}}$ Trig. syst.',
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
    "jesRelativePtBB":    r'JES $p_{\mathrm{T}}$ BB',
    "jesRelativePtEC1":   r'JES $p_{\mathrm{T}}$ EC1',
    "jesRelativePtEC2":   r'JES $p_{\mathrm{T}}$ EC2',
    "jesRelativePtHF":    r'JES $p_{\mathrm{T}}$ HF',
    "jesRelativeBal":     r'JES Bal',
    "jesRelativeFSR":     r'JES ISR+FSR',
    "jesRelativeStatFSR": r'JES ISR+FSR stat',
    "jesRelativeStatEC":  r'JES EC stat',
    "jesRelativeStatHF":  r'JES HF stat',
    "jesPileUpDataMC":    r'JES PU Data/MC',
    "jesPileUpPtRef":     r'JES PU $p_{\mathrm{T}}$ ref',
    "jesPileUpPtBB":      r'JES PU $p_{\mathrm{T}}$ BB',
    "jesPileUpPtEC1":     r'JES PU $p_{\mathrm{T}}$ EC1',
    "jesPileUpPtEC2":     r'JES PU $p_{\mathrm{T}}$ EC2',
    "jesPileUpPtHF":      r'JES PU $p_{\mathrm{T}}$ HF',
    "unclust":            r'Unclust En',
    "d1k_qcd":            r'$\delta^{(1)}K_{\mathrm{QCD}}$',
    "d2k_qcd":            r'$\delta^{(1)}K_{\mathrm{QCD}}$',
    "d3k_qcd":            r'$\delta^{(1)}K_{\mathrm{QCD}}$',
    "d1k_ew":             r'$\delta^{(1)}\kappa_{\mathrm{EW}}$',
    "d2k_ew_z":           r'$\delta^{(2)}\kappa_{\mathrm{EW}}^{\mathrm{Z}}$',
    "d2k_ew_w":           r'$\delta^{(2)}\kappa_{\mathrm{EW}}^{\mathrm{W}}$',
    "d3k_ew_z":           r'$\delta^{(3)}\kappa_{\mathrm{EW}}^{\mathrm{Z}}$',
    "d3k_ew_w":           r'$\delta^{(3)}\kappa_{\mathrm{EW}}^{\mathrm{W}}$',
    "dk_mix":             r'$\delta K_{\mathrm{mix}}$',
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
sample_names.update({"lhePdf{}".format(i): r'PDF_{'+str(i)+'}' for i in range(102)})
sample_names.update({"lheScale{}".format(i): r'Scale_{'+str(i)+'}' for i in range(9)})

axis_label = {
    "METnoX_pt": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)',
}
