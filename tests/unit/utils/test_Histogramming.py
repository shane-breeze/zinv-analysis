import pytest
import mock
import os
import copy
import numpy as np
import awkward as awk
import pandas as pd
import operator

from zinv.utils.Histogramming import Histograms
from zinv.utils.Lambda import Lambda

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.config = mock.MagicMock()

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return Histograms()

@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "region":   [
                "Monojet", "Monojet", "Monojet", "Monojet",
                "SingleElectron", "SingleElectron", "SingleElectron", "SingleElectron",
            ],
            "bin0_low": [0., 1., 2., 3., 0., 1., 2., 3.],
            "bin0_upp": [1., 2., 3., 4., 1., 2., 3., 4.],
            "count":    [0, 3, 5, 1, 0, 0, 0, 0],
            "yield":    [0.0, 0.5, 0.6, 0.4, 0., 0., 0., 0.],
            "variance": [0., 0.9, 0.3, 0.4, 0., 0., 0., 0.],
        },
        columns = ["region", "bin0_low", "bin0_upp", "count", "yield", "variance"],
    ).set_index(["region", "bin0_low", "bin0_upp"])

def test_histograms_init(module):
    assert module.histograms is None
    assert module.configs == []
    assert module.lambda_functions == {}
    assert module.binning == {}

configs = [
    {
        "name": "Jet_pt",
        "dataset": "MET",
        "region": "Monojet",
        "weight": "ev: ev.Weight_{dataset}_{cutflow}_{datamc}",
        "weightname": "nominal",
        "selection": ["ev: ev.Monojet_Cutflow"],
        "variables": ["ev: ev.Jet_pt"],
        "bins": [[0., 50., 100., 150., 200.]],
    }, {
        "name": ["Jet_pt", "Jet_eta"],
        "dataset": "SingleElectron",
        "region": "SingleElectron",
        "weight": "ev: ev.Weight_{dataset}_{cutflow}_{datamc}",
        "weightname": "nominal",
        "selection": ["ev: ev.SingleElectron_Cutflow"],
        "variables": ["ev: ev.Jet_pt", "ev: ev.Jet_eta"],
        "bins": [
            [0., 50., 100., 150., 200.],
            [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
        ],
    },
]
parents = ["DYJetsToEE", "DYJetsToMuMu", "DYJetsToTauTau"]
selection = {
    "DJetsToEE": ["ev: ev.LeptonIsElectron"],
    "DJetsToMuMu": ["ev: ev.LeptonIsMuon"],
    "DJetsToTauTau": ["ev: ev.LeptonIsTau"],
}

@pytest.mark.parametrize("configs", (configs,))
def test_histograms_extend(module, configs):
    print(configs)
    module.extend(configs)
    assert module.configs == configs
    module.binning["Jet_pt"] == [0., 50., 100., 150., 200.]
    module.binning["Jet_pt__Jet_eta"] == [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.]

@pytest.mark.parametrize(
    "parents,selection,configs", ([parents, selection, configs],)
)
def test_histograms_begin(module, event, parents, selection, configs):
    event.config.dataset.isdata = True
    event.config.dataset.name = "DYJetsToLL"
    module.configs = configs
    module.begin(event, parents, selection)

    assert module.isdata == True

    for idx, c in enumerate(module.full_configs):
        rc_idx = idx // len(parents)
        p_idx = idx % len(parents)
        rc = configs[rc_idx]
        p = parents[p_idx]

        assert c["name"] == rc["name"]
        assert c["dataset"] == rc["dataset"]
        assert c["region"] == rc["region"]
        assert c["weight"] == rc["weight"].format(
            dataset = rc["dataset"],
            cutflow = rc["region"],
            datamc = "Data",
        )
        assert c["weightname"] == rc["weightname"]
        assert c["selection"] == rc["selection"][0]
        assert c["variables"] == rc["variables"]
        assert c["bins"] == rc["bins"]

        assert c["process"] == p
        assert c["source"] == ""
        assert c["nsig"] == 0.

def test_histograms_end(module):
    module.clear_empties = mock.Mock()
    module.lambda_functions = "Something"
    module.end()

    assert module.clear_empties.called
    assert module.lambda_functions is None

def test_histograms_clear_empties_none(module):
    module.histograms = None
    assert module.clear_empties() is None
    assert module.histograms is None

def test_histograms_clear_empties(module, df):
    module.histograms = df
    module.clear_empties()
    assert module.histograms.equals(df.iloc[:4])

@pytest.mark.parametrize(
    "parents,selection,configs", ([parents, selection, configs],)
)
def test_histograms_event(module, event, df, parents, selection, configs):
    module.configs = configs
    module.generate_dataframe = mock.Mock(side_effect=lambda a, b: df.reset_index())
    module.begin(event, parents, selection)
    module.event(event)
    assert event.nsig == 0.
    assert event.source == ""
    assert ((module.histograms - (df*6)) < 1e-6).all().all()

@pytest.mark.parametrize(
    "parents,selection,configs,evattrs", (
        [
            parents,
            selection,
            configs,
            [
                ("Monojet_Cutflow", "bool8", [True, True, True, False, True, True, True]),
                ("Weight_MET_Monojet_Data", "int32", [1., 1., 0., 1., 1., 1., 1.]),
                ("Jet_pt", "float32", [50., 150., 250., 350., 55., 56., 57.]),
                ("SingleElectron_Cutflow", "bool8", [False, True, True, True, True, True, True]),
                ("Weight_SingleElectron_SingleElectron_Data", "int32", [1., 1., 1., 0., 1., 1., 1.]),
            ]
        ],
    )
)
def test_histograms_generate_dataframe(module, event, parents, selection, configs, evattrs):
    for label, dtype, array in evattrs:
        setattr(event, label, np.array(array, getattr(np, dtype)))
    event.size = len(evattrs[0][-1])
    event.config.dataset.isdata = True
    event.config.dataset.name = "DYJetsToLL"

    module.configs = configs
    module.begin(event, parents, selection)

    df = module.generate_dataframe(event, module.full_configs[0])

    odf = pd.DataFrame({
        "dataset": ["MET"]*3,
        "region": ["Monojet"]*3,
        "process": ["DYJetsToEE"]*3,
        "weight": ["nominal"]*3,
        "name": ["Jet_pt"]*3,
        "variable0": ["ev: ev.Jet_pt"]*3,
        "bin0_low": [50., 100., 150.],
        "bin0_upp": [100., 150., 200.],
        "count": [4., 1., 2.],
        "yield": [4., 1., 0.],
        "variance": [4., 1., 0.],
        "index": [1, 2, 3],
    }, columns=[
        "dataset", "region", "process", "weight", "name", "variable0",
        "bin0_low", "bin0_upp", "count", "yield", "variance", "index",
    ]).set_index("index")
    odf.index.name = None

    print(df)
    print(odf)

    string_cols = ["dataset", "region", "process", "weight", "name", "variable0"]
    val_cols = ["bin0_low", "bin0_upp", "count", "yield", "variance"]
    assert df[string_cols].equals(odf[string_cols])
    assert ((df[val_cols]-odf[val_cols])<1e-6).all().all()

def test_histograms_make_sparse_df(module, df):
    odf = df.iloc[1:4]
    df = module.make_sparse_df(df)
    print(odf)
    print(df)
    assert df.equals(odf)
