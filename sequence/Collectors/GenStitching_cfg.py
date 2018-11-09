import numpy as np
from utils.Colours import colours_dict

inf = np.infty

histogrammer_cfgs = [
    {
        "name": "LHE_Vpt",
        "categories": [("MET", "None")],
        "variables": ["ev: ev.LHE_Vpt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 101))+[inf]],
        "weights": [("xslumi", "ev: ev.Weight_XsLumi")],
    }, {
        "name": "Generator_scalePDF",
        "categories": [("MET", "None")],
        "variables": ["ev: ev.Generator_scalePDF"],
        "bins": [[-inf]+list(np.linspace(0., 4000., 101))+[inf]],
        "weights": [("xslumi", "ev: ev.Weight_XsLumi")],
    }, {
        "name": "GenPartBoson_pt",
        "categories": [("MET", "None")],
        "variables": ["ev: ev.GenPartBoson_pt"],
        "bins": [[-inf]+list(np.linspace(0., 1000., 101))+[inf]],
        "weights": [("xslumi", "ev: ev.Weight_XsLumi")],
    },
]

sample_colours = {
    "DYJetsToLL_Pt-0To50":    colours_dict["blue"],
    "DYJetsToLL_Pt-50To100":  colours_dict["green"],
    "DYJetsToLL_Pt-100To250": colours_dict["orange"],
    "DYJetsToLL_Pt-250To400": colours_dict["gold"],
    "DYJetsToLL_Pt-400To650": colours_dict["purple"],
    "DYJetsToLL_Pt-650ToInf": colours_dict["red"],

    "G1Jet_Pt-50To100":  colours_dict["blue"],
    "G1Jet_Pt-100To250": colours_dict["green"],
    "G1Jet_Pt-250To400": colours_dict["orange"],
    "G1Jet_Pt-400To650": colours_dict["gold"],
    "G1Jet_Pt-650ToInf": colours_dict["purple"],

    "WJetsToLNu_Pt-0To50":    colours_dict["blue"],
    "WJetsToLNu_Pt-50To100":  colours_dict["green"],
    "WJetsToLNu_Pt-100To250": colours_dict["orange"],
    "WJetsToLNu_Pt-250To400": colours_dict["gold"],
    "WJetsToLNu_Pt-400To600": colours_dict["purple"],
    "WJetsToLNu_Pt-600ToInf": colours_dict["red"],

    "ZJetsToNuNu_Pt-0To50":    colours_dict["blue"],
    "ZJetsToNuNu_Pt-50To100":  colours_dict["green"],
    "ZJetsToNuNu_Pt-100To250": colours_dict["orange"],
    "ZJetsToNuNu_Pt-250To400": colours_dict["gold"],
    "ZJetsToNuNu_Pt-400To650": colours_dict["purple"],
    "ZJetsToNuNu_Pt-650ToInf": colours_dict["red"],

    "QCD_Pt-15To30":     colours_dict["blue"],
    "QCD_Pt-30To50":     colours_dict["green"],
    "QCD_Pt-50To80":     colours_dict["orange"],
    "QCD_Pt-80To120":    colours_dict["gold"],
    "QCD_Pt-120To170":   colours_dict["purple"],
    "QCD_Pt-170To300":   colours_dict["pink"],
    "QCD_Pt-300To470":   colours_dict["violet"],
    "QCD_Pt-470To600":   colours_dict["red"],
    "QCD_Pt-600To800":   colours_dict["mint"],
    "QCD_Pt-800To1000":  colours_dict["yellow"],
    "QCD_Pt-1000To1400": colours_dict["blue"],
    "QCD_Pt-1400To1800": colours_dict["green"],
    "QCD_Pt-1800To2400": colours_dict["orange"],
    "QCD_Pt-2400To3200": colours_dict["gold"],
    "QCD_Pt-3200ToInf":  colours_dict["purple"],
}

sample_names = {
    'TTJets_Inclusive': r'$t\bar{t}$ + jets',

    'EWKWPlusToLNu2Jets':  r'$W_{l\nu}$ + 2 jets (VBF)',
    'EWKWMinusToLNu2Jets': r'$W_{l\nu}$ + 2 jets (VBF)',
    'EWKZToNuNu2Jets':     r'$Z_{\nu\nu}$ + 2 jets (VBF)',
    'EWKZToLL2Jets':       r'$Z_{ll}$ + 2 jets (VBF)',

    'WZTo2Q2Nu':   r'$W_{qq}Z_{\nu\nu}$',
    'WZTo2L2Q':    r'$W_{ll}Z_{qq}$',
    'WZTo3L1Nu':   r'$W_{l\nu}Z_{ll}$',
    'WZTo1L1Nu2Q': r'$W_{l\nu}Z_{qq}$',
    'ZZTo2Q2Nu':   r'$Z_{qq}Z_{\nu\nu}$',
    'ZZTo2L2Q':    r'$Z_{ll}Z_{qq}$',
    'ZZTo4Q':      r'$Z_{qq}Z_{qq}$',
    'WWTo2L2Nu':   r'$W_{l\nu}W_{l\nu}$',
    'WWTo4Q':      r'$W_{qq}W_{qq}$',
    'WWTo1L1Nu2Q': r'$W_{l\nu}W_{qq}$',
    'WZTo1L3Nu':   r'$W_{l\nu}Z_{\nu\nu}$',
    'WGToQQG':     r'$W_{qq}\gamma$',
    'WGToLNuG':    r'$W_{l\nu}\gamma$',
    'ZGToLLG':     r'$Z_{ll}\gamma$',
    'ZGToNuNuG':   r'$Z_{\nu\nu}\gamma$',
    'ZGToQQG':     r'$Z_{qq}\gamma$',
    'ZZTo4L':      r'$Z_{ll}Z_{ll}$',
    'ZZTo2L2Nu':   r'$Z_{ll}Z_{\nu\nu}$',

    'SingleTop_t-channel_top_InclusiveDecays':     r'Single Top (t-channel)',
    'SingleTop_t-channel_antitop_InclusiveDecays': r'Single Top (t-channel)',
    'SingleTop_tW_antitop_InclusiveDecays':        r'Single Top (tW)',
    'SingleTop_tW_top_InclusiveDecays':            r'Single Top (tW)',
    'SingleTop_s-channel_InclusiveDecays':         r'Single Top (s-channel)',

    "DYJetsToLL_Pt-0To50":    r'$Z/\gamma*_{ll}$ (0--50 GeV)',
    "DYJetsToLL_Pt-50To100":  r'$Z/\gamma*_{ll}$ (50--100 GeV)',
    "DYJetsToLL_Pt-100To250": r'$Z/\gamma*_{ll}$ (100--250 GeV)',
    "DYJetsToLL_Pt-250To400": r'$Z/\gamma*_{ll}$ (250--400 GeV)',
    "DYJetsToLL_Pt-400To650": r'$Z/\gamma*_{ll}$ (400--650 GeV)',
    "DYJetsToLL_Pt-650ToInf": r'$Z/\gamma*_{ll}$ (650+ GeV)',

    "G1Jet_Pt-50To100":  r'$\gamma+\rm{Jet}$ (50--100 GeV)',
    "G1Jet_Pt-100To250": r'$\gamma+\rm{Jet}$ (100--250 GeV)',
    "G1Jet_Pt-250To400": r'$\gamma+\rm{Jet}$ (250--400 GeV)',
    "G1Jet_Pt-400To650": r'$\gamma+\rm{Jet}$ (400--650 GeV)',
    "G1Jet_Pt-650ToInf": r'$\gamma+\rm{Jet}$ (650+ GeV)',

    "WJetsToLNu_Pt-0To50":    r'$W_{l\nu}$ (0--50 GeV)',
    "WJetsToLNu_Pt-50To100":  r'$W_{l\nu}$ (50--100 GeV)',
    "WJetsToLNu_Pt-100To250": r'$W_{l\nu}$ (100--250 GeV)',
    "WJetsToLNu_Pt-250To400": r'$W_{l\nu}$ (250--400 GeV)',
    "WJetsToLNu_Pt-400To600": r'$W_{l\nu}$ (400--600 GeV)',
    "WJetsToLNu_Pt-600ToInf": r'$W_{l\nu}$ (600+ GeV)',

    "ZJetsToNuNu_Pt-0To50":    r'$Z_{\nu\nu}$ (0--50 GeV)',
    "ZJetsToNuNu_Pt-50To100":  r'$Z_{\nu\nu}$ (50--100 GeV)',
    "ZJetsToNuNu_Pt-100To250": r'$Z_{\nu\nu}$ (100--250 GeV)',
    "ZJetsToNuNu_Pt-250To400": r'$Z_{\nu\nu}$ (250--400 GeV)',
    "ZJetsToNuNu_Pt-400To650": r'$Z_{\nu\nu}$ (400--650 GeV)',
    "ZJetsToNuNu_Pt-650ToInf": r'$Z_{\nu\nu}$ (650+ GeV)',

    "QCD_Pt-15To30":     r'QCD (15--30 GeV)',
    "QCD_Pt-30To50":     r'QCD (30--50 GeV)',
    "QCD_Pt-50To80":     r'QCD (50--80 GeV)',
    "QCD_Pt-80To120":    r'QCD (80--120 GeV)',
    "QCD_Pt-120To170":   r'QCD (120--170 GeV)',
    "QCD_Pt-170To300":   r'QCD (170--300 GeV)',
    "QCD_Pt-300To470":   r'QCD (300--470 GeV)',
    "QCD_Pt-470To600":   r'QCD (470--600 GeV)',
    "QCD_Pt-600To800":   r'QCD (600--800 GeV)',
    "QCD_Pt-800To1000":  r'QCD (800--1000 GeV)',
    "QCD_Pt-1000To1400": r'QCD (1000--1400 GeV)',
    "QCD_Pt-1400To1800": r'QCD (1400--1800 GeV)',
    "QCD_Pt-1800To2400": r'QCD (1800--2400 GeV)',
    "QCD_Pt-2400To3200": r'QCD (2400--3200 GeV)',
    "QCD_Pt-3200ToInf":  r'QCD (3200+ GeV)',
}

axis_label = {
    "LHE_Vpt": r'LHE $p_{T}(V)$ (GeV)',
    "Generator_scalePDF": r'LHE $Q^2$ ($\mathrm{GeV}^2$)',
    "GenPartBoson_pt": r'Gen $p_{T}(V)$ (GeV)',
}
