import numpy as np

inf = np.infty
pi = np.pi+0.00001

from Histogrammer_cfg import categories, sample_colours, sample_names, axis_label

histogrammer_cfgs = [
    {
        "name": ["LeadElectronSelection_pt", "LeadElectronSelection_etaSC"],
        "categories": [("SingleElectron", "SingleElectron")],
        "variables": ["ev: get_nth_object(ev.ElectronSelection.pt, 0, ev.size)",
                      "ev: get_nth_object(ev.ElectronSelection.eta, 0, ev.size)+get_nth_object(ev.ElectronSelection.deltaEtaSC, 0, ev.size)"],
        "bins": [[-inf, 0., 25., 28., 30., 32., 34., 36., 38., 40., 42., 44.,
                  46., 48., 50., 55., 60., 65., 70., 80., 90., 100., 120., 200.,
                  inf],
                 [-inf, -2.5, -2.0, -1.8, -1.566, -1.4442, -1.1, -0.8, -0.6, -0.4, -0.2, 0.0,
                  0.2, 0.4, 0.6, 0.8, 1.1, 1.4442, 1.566, 1.8, 2.0, 2.5, inf]],
        "weights": [("", "ev: ev.Weight_{dataset}"),
                    ("trigger", "ev: ev.Weight_{dataset}*(ev.Is{dataset}Triggered).astype(float)")],
    },
    # {
    #    "name": ["GenPartBoson_pt", "METnoX_pt"],
    #    "categories": categories,
    #    "variables": ["ev: ev.GenPartBoson_pt", "ev: ev.METnoX.pt"],
    #    "bins": [[-inf]+list(np.linspace(0, 1000, 51))+[inf],
    #             [-inf]+list(np.linspace(0, 1000, 51))+[inf]],
    #    "weights": [("", "ev: ev.Weight_{dataset}"),
    #                ("genWeight", "ev: ev.Weight_XsLumi"),
    #                ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    #}, {
    #    "name": ["GenPartBoson_phi", "METnoX_phi"],
    #    "categories": categories,
    #    "variables": ["ev: ev.GenPartBoson_phi", "ev: ev.METnoX.phi"],
    #    "bins": [[-inf]+list(np.linspace(-pi, pi, 51))+[inf],
    #             [-inf]+list(np.linspace(-pi, pi, 51))+[inf]],
    #    "weights": [("", "ev: ev.Weight_{dataset}"),
    #                ("genWeight", "ev: ev.Weight_XsLumi"),
    #                ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    #}, {
    #    "name": ["GenPartBoson_pt", "HMiss_pt"],
    #    "categories": categories,
    #    "variables": ["ev: ev.GenPartBoson.pt", "ev: ev.HMiss.pt"],
    #    "bins": [[-inf]+list(np.linspace(0, 1000, 51))+[inf],
    #             [-inf]+list(np.linspace(0, 1000, 51))+[inf]],
    #    "weights": [("", "ev: ev.Weight_{dataset}"),
    #                ("genWeight", "ev: ev.Weight_XsLumi"),
    #                ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    #}, {
    #    "name": ["GenPartBoson_eta", "HMiss_eta"],
    #    "categories": categories,
    #    "variables": ["ev: ev.GenPartBoson.eta", "ev: ev.HMiss.eta"],
    #    "bins": [[-inf]+list(np.linspace(-5., 5., 51))+[inf],
    #             [-inf]+list(np.linspace(-5., 5., 51))+[inf]],
    #    "weights": [("", "ev: ev.Weight_{dataset}"),
    #                ("genWeight", "ev: ev.Weight_XsLumi"),
    #                ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    #}, {
    #    "name": ["GenPartBoson_phi", "HMiss_phi"],
    #    "categories": categories,
    #    "variables": ["ev: ev.GenPartBoson.phi", "ev: ev.HMiss.phi"],
    #    "bins": [[-inf]+list(np.linspace(-pi, pi, 51))+[inf],
    #             [-inf]+list(np.linspace(-pi, pi, 51))+[inf]],
    #    "weights": [("", "ev: ev.Weight_{dataset}"),
    #                ("genWeight", "ev: ev.Weight_XsLumi"),
    #                ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    #{
    #    "name": ["METnoX_pt", "MinDPhiJ1234METnoX"],
    #    "categories": categories,
    #    "variables": ["ev: ev.METnoX.pt", "ev: ev.MinDPhiJ1234METnoX"],
    #    "bins":[[-inf]+list(np.linspace(0, 1000, 51))+[inf],
    #            [-inf]+list(np.linspace(0, pi, 51))+[inf]],
    #    "weights": [("", "ev: ev.Weight_{dataset}")],
    #},
]
