import pytest
import mock
import os
import numpy as np
import awkward as awk

from sequence.Readers import JecVariations

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.cache = {}
        self.attribute_variation_sources = ["jesTotal", "jerSF", "unclust", "jesAbsoluteStat"]
        self.config = mock.MagicMock()

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    toppath = os.path.abspath(os.environ["TOPDIR"])
    datapath = os.path.join(toppath, "data")
    return JecVariations(
        jes_unc_file = datapath + "/jecs/Summer16_23Sep2016V4_MC_UncertaintySources_AK4PFchs.txt",
        jer_sf_file = datapath + "/jecs/Spring16_25nsV10a_MC_SF_AK4PFchs.txt",
        jer_file = datapath + "/jecs/Spring16_25nsV10_MC_PtResolution_AK4PFchs.txt",
        apply_jer_corrections = True,
        jes_regex = "jes(?P<source>.*)",
        unclust_threshold = 15.,
    )

def test_jec_variations_begin(module, event):
    module.begin(event)
    assert all(t in ["Total", "AbsoluteStat"] for t in module.jes_sources)
    assert all(t in ["Total", "AbsoluteStat"] for t in event.JetSources)

params = [{
    "starts":      [0, 0, 1],
    "stopys":      [0, 1, 3],
    "jpt":         [16., 60., 70.],
    "jeta":        [-3.1, 0.5, 3.1],
    "jphi":        [0.1, 0.3, 0.5],
    "jgjidx":      [0, -1, 1],
    "gjpt":        [55., 65., 75.],
    "rho":         [0., 20., 50.],
    "met":         [200., 220., 240.],
    "mephi":       [0.9, 0.1, -0.7],
    "jptres":      [0.456227396877664, 0.131449148093119, 0.179063214239771],
    "jjersf":      [0.200500000000000, 1.271589764166470, 0.976571428571429],
    "jjersfdown":  [1.519638403990030, -0.061708613189990, 0.009142773551785],
    "jjersfup":    [-1., 0.050076915810078, -0.009142773551785],
    "new_met":     [200., 232.792, 232.10981279053],
    "new_mephi":   [0.9, 0.1, -0.75251459154],
    "jjestotdown": [-0.07725, -0.0144, -0.05163],
    "jjestotup":   [0.07725, 0.0144, 0.05163]
}]
@pytest.mark.parametrize(
    ",".join(params[0].keys()), (
        [ps[k] for k in params[0].keys()]
        for ps in params
    )
)
def test_jec_variations_event(
    module, event,
    starts, stopys, jpt, jeta, jphi, jgjidx, gjpt, rho, met, mephi,
    jptres, jjersf, jjersfdown, jjersfup, new_met, new_mephi,
    jjestotdown, jjestotup,
):
    def norm(*args, **kwargs):
        return 0.5
    np.random.normal = mock.Mock(side_effect=norm)
    event.config.dataset.idx = 2

    jet_pt = awk.JaggedArray(
        starts, stopys,
        np.array(jpt, dtype=np.float32),
    )
    jet_eta = awk.JaggedArray(
        starts, stopys,
        np.array(jeta, dtype=np.float32),
    )
    rho = np.array(rho, dtype=np.float32)

    jet_genjetidx = awk.JaggedArray(
        starts, stopys,
        np.array(jgjidx, dtype=np.int32),
    )
    genjet_pt = awk.JaggedArray(
        starts, stopys,
        np.array(gjpt, dtype=np.float32),
    )
    met_pt = np.array(met, dtype=np.float32)
    met_phi = np.array(mephi, dtype=np.float32)
    jet_phi = awk.JaggedArray(
        starts, stopys,
        np.array(jphi, dtype=np.float32),
    )

    event.Jet_pt = jet_pt
    event.Jet_eta = jet_eta
    event.fixedGridRhoFastjetAll = rho

    event.Jet_genJetIdx = jet_genjetidx
    event.GenJet_pt = genjet_pt
    event.MET_pt = met_pt
    event.MET_phi = met_phi
    event.Jet_phi = jet_phi

    module.begin(event)
    module.event(event)

    assert np.allclose(
        event.Jet_ptResolution.content,
        awk.JaggedArray(
            starts, stopys,
            np.array(jptres, dtype=np.float32),
        ).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjerSF.content,
        awk.JaggedArray(
            starts, stopys,
            np.array(jjersf, dtype=np.float32),
        ).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjerSFDown.content,
        awk.JaggedArray(
            starts, stopys,
            np.array(jjersfdown, dtype=np.float32),
        ).content,
        rtol=1e-5, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjerSFUp.content,
        awk.JaggedArray(
            starts, stopys,
            np.array(jjersfup, dtype=np.float32),
        ).content,
        rtol=1e-5, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_pt.content,
        awk.JaggedArray(
            starts, stopys,
            np.array(np.array(jpt)*np.array(jjersf), dtype=np.float32),
        ).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.MET_pt,
        np.array(new_met, dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.MET_phi,
        np.array(new_mephi, dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjesTotalUp.content,
        awk.JaggedArray(
            starts, stopys,
            np.array(jjestotup, dtype=np.float32),
        ).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjesTotalDown.content,
        awk.JaggedArray(
            starts, stopys,
            np.array(jjestotdown, dtype=np.float32),
        ).content,
        rtol=1e-6, equal_nan=True,
    )
