import numpy as np
import matplotlib.pyplot as plt
from utils.Colours import colours_dict

inf = np.infty
pi = np.pi+0.00001

# dataset-cutflows split into regions
muon_mu_categories = [("SingleMuon", "SingleMuon_noMETTrigger"), ("SingleMuon", "SingleMuon_METTrigger")]
dimuon_mu_categories = [("SingleMuon", "DoubleMuon_METTrigger"), ("SingleMuon", "DoubleMuon_noMETTrigger")]
trimuon_mu_categories = [("SingleMuon", "TripleMuon_METTrigger"), ("SingleMuon", "TripleMuon_noMETTrigger")]
quadmuon_mu_categories = [("SingleMuon", "QuadMuon_METTrigger"), ("SingleMuon", "QuadMuon_noMETTrigger")]
ele_categories = [("SingleElectron", "SingleElectron_METTrigger"), ("SingleElectron", "SingleElectron_noMETTrigger")]
diele_categories = [("SingleElectron", "DoubleElectron_METTrigger"), ("SingleElectron", "DoubleElectron_noMETTrigger")]
triele_categories = [("SingleElectron", "TripleElectron_METTrigger"), ("SingleElectron", "TripleElectron_noMETTrigger")]
quadele_categories = [("SingleElectron", "QuadElectron_METTrigger"), ("SingleElectron", "QuadElectron_noMETTrigger")]

mu_categories = muon_mu_categories + dimuon_mu_categories + trimuon_mu_categories + quadmuon_mu_categories
ele_categories = ele_categories + diele_categories + triele_categories + quadele_categories

monojet_variations = [("nominal", "ev: ev.Weight_{dataset}")]

histogrammer_cfgs = [
    {
        "name": "METnoX_pt",
        "categories": mu_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1500., 301))+[inf]],
        "weights": [("nominal", "ev: ev.Weight_{dataset}")]
    }, {
        "name": "MET_pt",
        "categories": ele_categories,
        "variables": ["ev: ev.MET_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1500., 301))+[inf]],
        "weights": [("nominal", "ev: ev.Weight_{dataset}")]
    },
]

sample_colours = {
    "nominal": "black",
}

sample_names = {
    "nominal": r'Nominal',
    "znunu":   r'$Z_{\nu\nu}$+jets',
    "wlnu":    r'$W_{l\nu}$+jets',
    "bkg":     r'Bkg',
    "qcd":     r'QCD',
    "zmumu":   r'$Z/\gamma^{*}_{\mu\mu}$+jets',
    "zee":     r'$Z/\gamma^{*}_{e e}$+jets',
}

axis_label = {
    "METnoX_pt": r'$p_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)',
    "MET_pt": r'$p_{\mathrm{T,PF}}^{\mathrm{miss}}$ (GeV)',
}
