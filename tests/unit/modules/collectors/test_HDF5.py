import pytest
import mock

from zinv.modules.collectors import HDF5Reader

def test_init():
    obj = HDF5Reader()
    assert obj.measure_timing == False
    assert obj.attribute_timing == {}

def test_merge():
    attr_timing1 = {"k1": 10}
    attr_timing2 = {"k1": 5, "k2": 20}

    self = HDF5Reader()
    other = HDF5Reader()

    self.attribute_timing = attr_timing1
    other.attribute_timing = attr_timing2

    self.merge(other)
    assert self.attribute_timing == {"k1": 15, "k2": 20}

#def test_begin():
#    self = HDF5Reader()
#    self.hdf5_config_path = "dummy.yaml"
#
#    data = (
#        'name: "Events"'
#        'output: "outdir"'
#        'variations: []'
#        'attributes:'
#        '    Both:'
#        '        run: "ev, source, nsig: ev.run"'
#        '    MC:'
#        '        GenPartBoson_pt: "ev, source, nsig: ev.GenPartBoson_pt(ev, source, nsig)"'
#        '
