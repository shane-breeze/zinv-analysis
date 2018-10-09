import numpy as np
from utils.Colours import colours_dict

inf = np.infty
pi = np.pi+0.00001

# dataset-cutflows split into regions
monojet_categories = [("MET", "Monojet"), ("MET", "MonojetSB"), ("MET", "MonojetSR"),
                      ("MET", "MonojetQCD"), ("MET", "MonojetQCDSB"), ("MET", "MonojetQCDSR")]

muon_categories = [("MET", "SingleMuon"), ("MET", "SingleMuonSB"), ("MET", "SingleMuonSR"),
                   ("SingleMuon", "SingleMuon"), ("SingleMuon", "SingleMuonSB"), ("SingleMuon", "SingleMuonSR")]
dimuon_categories = [("MET", "DoubleMuon"), ("MET", "DoubleMuonSB"), ("MET", "DoubleMuonSR"),
                     ("SingleMuon", "DoubleMuon"),("SingleMuon", "DoubleMuonSB"), ("SingleMuon", "DoubleMuonSR")]

ele_categories = [("MET", "SingleElectron"), ("MET", "SingleElectronSB"), ("MET", "SingleElectronSR")]
diele_categories = [("MET", "DoubleElectron"), ("MET", "DoubleElectronSB"), ("MET", "DoubleElectronSR")]

categories = monojet_categories + muon_categories + dimuon_categories + \
                                  ele_categories + diele_categories

monojet_variations = [
    ("nominal",       "ev: ev.Weight_{dataset}"),
    ("pileupUp",      "ev: ev.Weight_{dataset}*ev.Weight_pileupUp"),
    ("pileupDown",    "ev: ev.Weight_{dataset}*ev.Weight_pileupDown"),
    ("metTrigSFUp",   "ev: ev.Weight_{dataset}*ev.Weight_metTrigSFUp"),
    ("metTrigSFDown", "ev: ev.Weight_{dataset}*ev.Weight_metTrigSFDown"),
]

muon_variations = monojet_variations + [
    ("muonIdUp",      "ev: ev.Weight_{dataset}*ev.Weight_muonIdUp"),
    ("muonIdDown",    "ev: ev.Weight_{dataset}*ev.Weight_muonIdDown"),
    ("muonIsoUp",     "ev: ev.Weight_{dataset}*ev.Weight_muonIsoUp"),
    ("muonIsoDown",   "ev: ev.Weight_{dataset}*ev.Weight_muonIsoDown"),
    ("muonTrackUp",   "ev: ev.Weight_{dataset}*ev.Weight_muonTrackUp"),
    ("muonTrackDown", "ev: ev.Weight_{dataset}*ev.Weight_muonTrackDown"),
    ("muonTrigUp",    "ev: ev.Weight_{dataset}*ev.Weight_muonTrigUp"),
    ("muonTrigDown",  "ev: ev.Weight_{dataset}*ev.Weight_muonTrigDown"),
]

ele_variations = monojet_variations

histogrammer_cfgs = [
    {
        "name": "METnoX_pt",
        "categories": monojet_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 41))+[inf]],
        "weights": monojet_variations,
    }, {
        "name": "METnoX_pt",
        "categories": muon_categories + dimuon_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 41))+[inf]],
        "weights": muon_variations,
    }, {
        "name": "METnoX_pt",
        "categories": ele_categories + diele_categories,
        "variables": ["ev: ev.METnoX_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 41))+[inf]],
        "weights": ele_variations,
    }
] + [
    {
        "name": "METnoX_pt",
        "categories": [(d, c+variation) for d, c in categories],
        "variables": ["ev: ev.METnoX_pt{}".format(variation)],
        "bins": [[-inf]+list(np.linspace(0., 1000., 41))+[inf]],
        "weights": [(variation, "ev: ev.Weight_{dataset}")],
    } for variation in ["jesUp", "jesDown",
                        "jerUp", "jerDown",
                        "unclustUp", "unclustDown"]
]

sample_colours = {
    "":              "black",
    "pileupUp":      colours_dict["gray"],
    "pileupDown":    colours_dict["gray"],
    "metTrigSFUp":   colours_dict["blue"],
    "metTrigSFDown": colours_dict["blue"],
    "muonIdUp":      colours_dict["green"],
    "muonIdDown":    colours_dict["green"],
    "muonIsoUp":     colours_dict["orange"],
    "muonIsoDown":   colours_dict["orange"],
    "muonTrackUp":   colours_dict["gold"],
    "muonTrackDown": colours_dict["gold"],
    "muonTrigUp":    colours_dict["purple"],
    "muonTrigDown":  colours_dict["purple"],
    "jesUp":         colours_dict["red"],
    "jesDown":       colours_dict["red"],
    "jerUp":         colours_dict["pink"],
    "jerDown":       colours_dict["pink"],
    "unclustUp":     colours_dict["violet"],
    "unclustDown":   colours_dict["violet"],
}

sample_names = {
    "":              "Nominal",
    "pileupUp":      "Pileup",
    "pileupDown":    "Pileup",
    "metTrigSFUp":   r'$E_{T}^{miss}$ Trig',
    "metTrigSFDown": r'$E_{T}^{miss}$ Trig',
    "muonIdUp":      "Muon ID",
    "muonIdDown":    "Muon ID",
    "muonIsoUp":     "Muon Iso",
    "muonIsoDown":   "Muon Iso",
    "muonTrackUp":   "Muon Track",
    "muonTrackDown": "Muon Track",
    "muonTrigUp":    "Muon Trig",
    "muonTrigDown":  "Muon Trig",
    "jesUp":         "JES",
    "jesDown":       "JES",
    "jerUp":         "JER",
    "jerDown":       "JER",
    "unclustUp":     "Unclust. En.",
    "unclustDown":   "Unclust. En.",
}

axis_label = {
    "METnoX_pt": r'$E_{T}^{miss}$ (GeV)',
}
