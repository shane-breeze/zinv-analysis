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
    "nominal":   "black",
    "mcstat":    colours_dict["gray"],
    "pileup":    colours_dict["mint"],
    "metTrigSF": colours_dict["blue"],
    "muonId":    colours_dict["green"],
    "muonIso":   colours_dict["orange"],
    "muonTrack": colours_dict["gold"],
    "muonTrig":  colours_dict["purple"],
    "jes":       colours_dict["red"],
    "jer":       colours_dict["pink"],
    "unclust":   colours_dict["violet"],
}

sample_names = {
    "nominal":   "Nominal",
    "mcstat":    "MC stat.",
    "pileup":    "Pileup",
    "metTrigSF": r'$E_{T}^{miss}$ Trig',
    "muonId":    "Muon ID",
    "muonIso":   "Muon Iso",
    "muonTrack": "Muon Track",
    "muonTrig":  "Muon Trig",
    "jes":       "JES",
    "jer":       "JER",
    "unclust":   "Unclust. En.",
}

axis_label = {
    "METnoX_pt": r'$E_{T}^{miss}$ (GeV)',
}
