import pytest
import mock
import numpy as np
import awkward as awk

from zinv.utils.Geometry import (
    BoundPhi,
    DeltaR2,
    RadToCart2D,
    CartToRad2D,
    PartCoorToCart3D,
    CartToPartCoor3D,
    LorTHPMToXYZE,
    LorXYZEToTHPM,
)

@pytest.mark.parametrize("phi,ophi", ([
    1., 1.
], [
    np.pi+1., 1.-np.pi
], [
    -1.-np.pi, -1+np.pi
], [
    [1., np.pi+1.], [1., 1.-np.pi]
]))
def test_BoundPhi(phi, ophi):
    assert np.allclose(BoundPhi(phi), ophi, rtol=1e-5, equal_nan=True)
