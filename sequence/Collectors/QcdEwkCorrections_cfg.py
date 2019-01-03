import numpy as np
from utils.Colours import colours_dict
inf = np.infty

categories = [("MET", "None"), ("MET", "Monojet"), ("MET", "MonojetQCD"),
              ("MET", "SingleMuon"), ("SingleMuon", "SingleMuon"),
              ("MET", "DoubleMuon"), ("SingleMuon", "DoubleMuon"),
              ("SingleElectron", "SingleElectron"),
              ("SingleElectron", "DoubleElectron")]

histogrammer_cfgs = [
    {
        "name": "GenPartBoson_pt",
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson_pt"],
        "bins": [[-inf]+list(np.linspace(0., 2000., 51))+[inf]],
        "weights": [("nominal", "ev: ev.Weight_XsLumi")],
    }, {
        "name": "GenPartBoson_pt_corrected",
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson_pt"],
        "bins": [[-inf]+list(np.linspace(0., 2000., 51))+[inf]],
        "weights": [("nominal", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    },
]

sample_colours = {
    "MET":              "black",
    "SingleMuon":       "black",
    "SingleElectron":   "black",
    "ZJetsToNuNu":      colours_dict["blue"],
    "WJetsToLNu":       colours_dict["green"],
    "WJetsToENu":       colours_dict["lgreen"],
    "WJetsToMuNu":      colours_dict["green"],
    "WJetsToTauLNu":    colours_dict["teal"],
    "WJetsToTauHNu":    colours_dict["teal"],
    "Diboson":          colours_dict["orange"],
    "DYJetsToLL":       colours_dict["gold"],
    "DYJetsToEE":       colours_dict["gold1"],
    "DYJetsToMuMu":     colours_dict["gold2"],
    "DYJetsToTauLTauL": colours_dict["gold3"],
    "DYJetsToTauLTauH": colours_dict["gold3"],
    "DYJetsToTauHTauH": colours_dict["gold3"],
    "EWKV2Jets":        colours_dict["purple"],
    "SingleTop":        colours_dict["pink"],
    "TTJets":           colours_dict["violet"],
    "QCD":              colours_dict["red"],
    "G1Jet":            colours_dict["mint"],
    "VGamma":           colours_dict["yellow"],
    "Minor":            colours_dict["gray"],
}

sample_names = {
    "MET":              r'MET',
    "SingleMuon":       r'Single Muon',
    "SingleElectron":   r'Single Electron',
    "ZJetsToNuNu":      r'$Z_{\nu\nu}$+jets',
    "WJetsToLNu":       r'$W_{l\nu}$+jets',
    "WJetsToENu":       r'$W_{e\nu}$+jets',
    "WJetsToMuNu":      r'$W_{\mu\nu}$+jets',
    "WJetsToTauLNu":    r'$W_{\tau_{l}\nu}$+jets',
    "WJetsToTauHNu":    r'$W_{\tau_{h}\nu}$+jets',
    "Diboson":          r'Diboson',
    "DYJetsToLL":       r'$Z/\gamma^{*}_{ll}$+jets',
    "DYJetsToEE":       r'$Z/\gamma^{*}_{ee}$+jets',
    "DYJetsToMuMu":     r'$Z/\gamma^{*}_{\mu\mu}$+jets',
    "DYJetsToTauLTauL": r'$Z/\gamma^{*}_{\tau_{l}\tau_{l}}$+jets',
    "DYJetsToTauLTauH": r'$Z/\gamma^{*}_{\tau_{l}\tau_{h}}$+jets',
    "DYJetsToTauHTauH": r'$Z/\gamma^{*}_{\tau_{h}\tau_{h}}$+jets',
    "EWKV2Jets":        r'VBS',
    "SingleTop":        r'Single Top',
    "TTJets":           r'$t\bar{t}$+jets',
    "QCD":              r'Multijet',
    "G1Jet":            r'$\gamma$+jets',
    "VGamma":           r'$V+\gamma$',
    "Minor":            r'Minors',
}

axis_label = {
    "GenPartBoson_pt": r'$p_{T}(V)$ (GeV)',
    "GenPartBoson_pt_corrected": r'Corrected $p_{T}(V)$ (GeV)',
}
