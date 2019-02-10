import numpy as np
import matplotlib.pyplot as plt
from utils.Colours import colours_dict

inf = np.infty
pi = np.pi+0.00001

# dataset-cutflows split into regions
categories = [("MET", "None"), ("MET", "DoubleMuon")]

variations = [
    ("nominal", "ev: ev.Weight_{dataset}"),
] + [
    ("lhePdf{}".format(i), "ev: ev.Weight_{dataset}"+"*ev.LHEPdfWeight[:,{0}] if {0} < ev.nLHEPdfWeight[0] else np.full(ev.size, np.nan)".format(i))
    for i in range(0,110)
] + [
    ("lheScale{}".format(i), "ev: ev.Weight_{dataset}"+"*ev.LHEScaleWeight[:,{0}] if {0} < ev.nLHEScaleWeight[0] else np.full(ev.size, np.nan)".format(i))
    for i in (0,1,3,5,7,8)
]

histogrammer_cfgs = [
    {
        "name": "GenPartBoson_mass",
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson_mass"],
        "bins": [[-inf]+list(np.linspace(50., 150., 201))+[inf]],
        "weights": variations,
    }, {
        "name": "GenPartBoson_pt",
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson_mass"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 101))+[inf]],
        "weights": variations,
    },
]

sample_colours = {
    "DYJetsToLL": "black",
    "ZJetsToLL":  colours_dict["red"],
    "GStarJetsToLL": colours_dict["green"],
    "Z+GStarJetsToLL": colours_dict["blue"],
}

sample_names = {
    "DYJetsToLL":      r'$\sigma(\mathrm{Z}/\gamma^{*})$',
    "ZJetsToLL":       r'$\sigma(\mathrm{Z})$',
    "GStarJetsToLL":   r'$\sigma(\gamma^{*})$',
    "Z+GStarJetsToLL": r'$\sigma(\mathrm{Z})+\sigma(\gamma^{*})$',
}

axis_label = {
    "GenPartBoson_mass": r'$m_{ll}$ (GeV)',
    "GenPartBoson_pt": r'$p_{\mathrm{T}}(\mathrm{Z})$ (GeV)',
}
