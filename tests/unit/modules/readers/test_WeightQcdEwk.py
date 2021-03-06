import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.modules.readers import WeightQcdEwk

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.cache = {}

        self.config = mock.MagicMock()

    def register_function(self, event, name, function):
        self.__dict__[name] = function

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return WeightQcdEwk(
        input_paths = {
            "ZJetsToNuNu": ("http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/qcd_ewk/vvj.dat", "vvj_pTV_{}"),
            "WJetsToLNu":  ("http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/qcd_ewk/evj.dat", "evj_pTV_{}"),
            "DYJetsToLL":  ("http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/qcd_ewk/eej.dat", "eej_pTV_{}"),
        },
        underflow = True,
        overflow = True,
        formula = "((K_NNLO + d1k_qcd*d1K_NNLO + d2k_qcd*d2K_NNLO + d3k_qcd*d3K_NNLO)"\
                  "*(1 + kappa_EW + d1k_ew*d1kappa_EW + isz*(d2k_ew_z*d2kappa_EW + d3k_ew_z*d3kappa_EW)"\
                                                     "+ isw*(d2k_ew_w*d2kappa_EW + d3k_ew_w*d3kappa_EW))"\
                  "+ dk_mix*dK_NNLO_mix)"\
                  "/(K_NLO + d1k_qcd*d1K_NLO + d2k_qcd*d2K_NLO + d3k_qcd*d3K_NLO)",
        params = ["K_NLO", "d1K_NLO", "d2K_NLO", "d3K_NLO", "K_NNLO", "d1K_NNLO",
                  "d2K_NNLO", "d3K_NNLO", "kappa_EW", "d1kappa_EW", "d2kappa_EW",
                  "d3kappa_EW", "dK_NNLO_mix"],
        variation_names = ["d1k_qcd", "d2k_qcd", "d3k_qcd", "d1k_ew", "d2k_ew_z",
                           "d2k_ew_w", "d3k_ew_z", "d3k_ew_w", "dk_mix"],
    )

def test_weightqcdewk_init(module):
    assert module.input_paths == {
        "ZJetsToNuNu": ("http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/qcd_ewk/vvj.dat", "vvj_pTV_{}"),
        "WJetsToLNu":  ("http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/qcd_ewk/evj.dat", "evj_pTV_{}"),
        "DYJetsToLL":  ("http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/qcd_ewk/eej.dat", "eej_pTV_{}"),
    }
    assert module.underflow == True
    assert module.overflow == True
    assert module.formula == "((K_NNLO + d1k_qcd*d1K_NNLO + d2k_qcd*d2K_NNLO + d3k_qcd*d3K_NNLO)"\
                  "*(1 + kappa_EW + d1k_ew*d1kappa_EW + isz*(d2k_ew_z*d2kappa_EW + d3k_ew_z*d3kappa_EW)"\
                                                     "+ isw*(d2k_ew_w*d2kappa_EW + d3k_ew_w*d3kappa_EW))"\
                  "+ dk_mix*dK_NNLO_mix)"\
            "/(K_NLO + d1k_qcd*d1K_NLO + d2k_qcd*d2K_NLO + d3k_qcd*d3K_NLO)"
    assert module.params == [
        "K_NLO", "d1K_NLO", "d2K_NLO", "d3K_NLO", "K_NNLO", "d1K_NNLO",
        "d2K_NNLO", "d3K_NNLO", "kappa_EW", "d1kappa_EW", "d2kappa_EW",
        "d3kappa_EW", "dK_NNLO_mix",
    ]
    assert module.variation_names == [
        "d1k_qcd", "d2k_qcd", "d3k_qcd", "d1k_ew", "d2k_ew_z",
        "d2k_ew_w", "d3k_ew_z", "d3k_ew_w", "dk_mix",
    ]

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "parent": "ZJetsToNuNu",
            "underflow": True,
            "overflow": True,
        }, {
            "result": True,
        }], [{
            "parent": "ZJetsToNuNu",
            "underflow": False,
            "overflow": False,
        }, {
            "result": True,
        }], [{
            "parent": "WJetsToLNu",
            "underflow": True,
            "overflow": True,
        }, {
            "result": True,
        }], [{
            "parent": "DYJetsToLL",
            "underflow": True,
            "overflow": True,
        }, {
            "result": True,
        }], [{
            "parent": "Other",
            "underflow": True,
            "overflow": True,
        }, {
            "result": True,
        }],
    )
)
def test_weightqcdewk_begin(module, event, inputs, outputs):
    event.config.dataset.parent = inputs["parent"]
    module.underflow = inputs["underflow"]
    module.overflow = inputs["overflow"]
    module.begin(event)
    assert hasattr(module, "input_df") == outputs["result"]

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "",
            "nsig": 0.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.071991350069300, 1.085072600134150, 1.069131960784340, 1.030109454980090, 0.943911876078431, 0.864166469122489, 0.774704840108300, 0.721218007769897, 0.721218007769897],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d1k_qcd",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.030696018270610, 1.047052839720330, 1.028880006209720, 0.983430952350240, 0.893472013440240, 0.812196867984426, 0.724960805696453, 0.669169906367341, 0.669169906367341],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d1k_qcd",
            "nsig": -1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.118116532226270, 1.127671526399130, 1.114356625413100, 1.083723429282010, 1.003526722622880, 0.927100703776154, 0.837949066500746, 0.791748390727572, 0.791748390727572],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d2k_qcd",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.117909027020050, 1.127140552545750, 1.112142453111700, 1.073887137958910, 0.958234879552157, 0.841916736146612, 0.733627645604361, 0.673274194374568, 0.673274194374568],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d3k_qcd",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.082327160627540, 1.092411784236760, 1.073723566545470, 1.031150128579600, 0.942417841941239, 0.860581484315055, 0.768699259163798, 0.712640895377803, 0.712640895377803],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d1k_ew",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.071991749507280, 1.085073070394590, 1.069132821120370, 1.030186252380140, 0.945344716388350, 0.871445952688314, 0.801250740529584, 0.774453223619960, 0.774453223619960],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d2k_ew_z",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.072396129258980, 1.085415071405440, 1.069300713259830, 1.032380364575490, 0.951036728220107, 0.877034497455687, 0.795210818772591, 0.747938032557757, 0.747938032557757],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d2k_ew_w",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.071991350069300, 1.085072600134150, 1.069131960784340, 1.030109454980090, 0.943911876078431, 0.864166469122489, 0.774704840108300, 0.721218007769897, 0.721218007769897],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d3k_ew_z",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.072039280090420, 1.085161872164570, 1.069536581692570, 1.031869050982240, 0.950568389353791, 0.879032083520971, 0.801688322421896, 0.753732857417780, 0.753732857417780],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d3k_ew_w",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.071991350069300, 1.085072600134150, 1.069131960784340, 1.030109454980090, 0.943911876078431, 0.864166469122489, 0.774704840108300, 0.721218007769897, 0.721218007769897],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "dk_mix",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.071739594941190, 1.084841705104440, 1.069235891413690, 1.031720721296600, 0.948853206957072, 0.872806553702445, 0.788247086111982, 0.739514507629186, 0.739514507629186],
        }], [{
            "parent": "Other",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "",
            "nsig": 0.,
        }, {
            "weight": [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        }], [{
            "parent": "ZJetsToNuNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "Other",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.071991350069300, 1.085072600134150, 1.069131960784340, 1.030109454980090, 0.943911876078431, 0.864166469122489, 0.774704840108300, 0.721218007769897, 0.721218007769897],
        }], [{
            "parent": "WJetsToLNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "",
            "nsig": 0.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.023410820320550, 1.024130192347730, 1.020314922339990, 0.990417237421081, 0.903637981081761, 0.819501618321920, 0.712597842165201, 0.588238663003707, 0.588238663003707],
        }], [{
            "parent": "WJetsToLNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d2k_ew_z",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.023410820320550, 1.024130192347730, 1.020314922339990, 0.990417237421081, 0.903637981081761, 0.819501618321920, 0.712597842165201, 0.588238663003707, 0.588238663003707],
        }], [{
            "parent": "WJetsToLNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d2k_ew_w",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.023589821835780, 1.024624881588750, 1.021715924807730, 0.994258370841088, 0.912723304594549, 0.834961499333771, 0.737826329444925, 0.624669558858385, 0.624669558858385],
        }], [{
            "parent": "WJetsToLNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d3k_ew_z",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.023410820320550, 1.024130192347730, 1.020314922339990, 0.990417237421081, 0.903637981081761, 0.819501618321920, 0.712597842165201, 0.588238663003707, 0.588238663003707],
        }], [{
            "parent": "WJetsToLNu",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d3k_ew_w",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.023481240006130, 1.024178658506250, 1.020342371632680, 0.990579502649820, 0.905652659464953, 0.825173382215209, 0.710119702126497, 0.522045582148113, 0.522045582148113],
        }], [{
            "parent": "DYJetsToLL",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "",
            "nsig": 0.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.057695056859910, 1.065163565548090, 1.040220306435990, 1.003716881754210, 0.929583009857246, 0.866283824910207, 0.798585450112642, 0.762354801796986, 0.762354801796986],
        }], [{
            "parent": "DYJetsToLL",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d2k_ew_z",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.057893818386340, 1.065452466733380, 1.041603936377610, 1.007252710288180, 0.937512118447112, 0.879286702498615, 0.818381967026762, 0.787824254431365, 0.787824254431365],
        }], [{
            "parent": "DYJetsToLL",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d2k_ew_w",
            "nsig": 0.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.057695056859910, 1.065163565548090, 1.040220306435990, 1.003716881754210, 0.929583009857246, 0.866283824910207, 0.798585450112642, 0.762354801796986, 0.762354801796986],
        }], [{
            "parent": "DYJetsToLL",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d3k_ew_z",
            "nsig": 1.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.057765538281910, 1.065258256901350, 1.040269795465800, 1.004102021154720, 0.934022109645626, 0.880831209866396, 0.832310598300416, 0.810421192941440, 0.810421192941440],
        }], [{
            "parent": "DYJetsToLL",
            "vpt": [0., 10., 20., 30., 50., 100., 200., 500., 1000., 2000., 5000., 10000.],
            "source": "d3k_ew_w",
            "nsig": 0.,
        }, {
            "weight": [1.000000000000000, 1.000000000000000, 1.000000000000000, 1.057695056859910, 1.065163565548090, 1.040220306435990, 1.003716881754210, 0.929583009857246, 0.866283824910207, 0.798585450112642, 0.762354801796986, 0.762354801796986],
        }],
    )
)
def test_weightqcdewk_event(module, event, inputs, outputs):
    event.config.dataset.parent = inputs["parent"]
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]
    event.size = len(inputs["vpt"])

    module.begin(event)

    vpt = np.array(inputs["vpt"], dtype=np.float32)
    event.GenPartBoson = mock.Mock(side_effect=lambda ev, attr: vpt)

    weight = event.WeightQcdEwk(event, event.source, event.nsig)
    oweight = np.array(outputs["weight"], dtype=np.float32)

    print(inputs["parent"])
    print(inputs["source"])
    print(inputs["nsig"])
    print(weight)
    print(oweight)
    assert np.allclose(weight, oweight, rtol=1e-6, equal_nan=True)
