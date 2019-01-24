import numpy as np

inf = np.infty
pi = np.pi+0.00001

from .Histogrammer_cfg import categories, sample_colours, sample_names, axis_label

histogrammer_cfgs = [
    {
        "name": ["GenPartBoson_pt", "METnoX_pt"],
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson_pt", "ev: ev.METnoX.pt"],
        "bins": [[-inf]+list(np.linspace(0, 1000, 51))+[inf],
                 [-inf]+list(np.linspace(0, 1000, 51))+[inf]],
        "weights": [("", "ev: ev.Weight_{dataset}"),
                    ("genWeight", "ev: ev.Weight_XsLumi"),
                    ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    }, {
        "name": ["GenPartBoson_phi", "METnoX_phi"],
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson_phi", "ev: ev.METnoX.phi"],
        "bins": [[-inf]+list(np.linspace(-pi, pi, 51))+[inf],
                 [-inf]+list(np.linspace(-pi, pi, 51))+[inf]],
        "weights": [("", "ev: ev.Weight_{dataset}"),
                    ("genWeight", "ev: ev.Weight_XsLumi"),
                    ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    }, {
        "name": ["GenPartBoson_pt", "HMiss_pt"],
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson.pt", "ev: ev.HMiss.pt"],
        "bins": [[-inf]+list(np.linspace(0, 1000, 51))+[inf],
                 [-inf]+list(np.linspace(0, 1000, 51))+[inf]],
        "weights": [("", "ev: ev.Weight_{dataset}"),
                    ("genWeight", "ev: ev.Weight_XsLumi"),
                    ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    }, {
        "name": ["GenPartBoson_eta", "HMiss_eta"],
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson.eta", "ev: ev.HMiss.eta"],
        "bins": [[-inf]+list(np.linspace(-5., 5., 51))+[inf],
                 [-inf]+list(np.linspace(-5., 5., 51))+[inf]],
        "weights": [("", "ev: ev.Weight_{dataset}"),
                    ("genWeight", "ev: ev.Weight_XsLumi"),
                    ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    }, {
        "name": ["GenPartBoson_phi", "HMiss_phi"],
        "categories": categories,
        "variables": ["ev: ev.GenPartBoson.phi", "ev: ev.HMiss.phi"],
        "bins": [[-inf]+list(np.linspace(-pi, pi, 51))+[inf],
                 [-inf]+list(np.linspace(-pi, pi, 51))+[inf]],
        "weights": [("", "ev: ev.Weight_{dataset}"),
                    ("genWeight", "ev: ev.Weight_XsLumi"),
                    ("nNLOEW", "ev: ev.Weight_XsLumi*ev.WeightQcdEwk")],
    }, {
        "name": ["METnoX_pt", "MinDPhiJ1234METnoX"],
        "categories": categories,
        "variables": ["ev: ev.METnoX.pt", "ev: ev.MinDPhiJ1234METnoX"],
        "bins":[[-inf]+list(np.linspace(0, 1000, 51))+[inf],
                [-inf]+list(np.linspace(0, pi, 51))+[inf]],
        "weights": [("", "ev: ev.Weight_{dataset}")],
    },
]
