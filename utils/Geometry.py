import numpy as np
from numpy import pi
from numba import njit, vectorize

@vectorize
def BoundPhi(phi):
    if phi >= pi:
        phi -= 2*pi
    elif phi < -pi:
        phi += 2*pi
    return phi

@njit
def DeltaR2(deta, dphi):
    return deta**2 + BoundPhi(dphi)**2

@njit
def RadToCart2D(r, phi):
    return r*np.cos(phi), r*np.sin(phi)

@njit
def CartToRad2D(x, y):
    return np.sqrt(x**2+y**2), BoundPhi(np.arctan2(y, x))

@njit
def PartCoorToCart3D(pt, eta, phi):
    x, y = RadToCart2D(pt, phi)
    z = pt*np.sinh(eta)
    return x, y, z

@njit
def CartToPartCoor3D(x, y, z):
    pt, phi = CartToRad2D(x, y)
    eta = np.arctanh(z/np.sqrt(z**2+pt**2))
    return pt, eta, phi

@njit
def LorTHPMToXYZE(t, h, p, m):
    x = t*np.cos(p)
    y = t*np.sin(p)
    z = t*np.sinh(h)
    e = np.sqrt(m**2 + t**2 + z**2)
    return x, y, z, e

@njit
def LorXYZEToTHPM(x, y, z, e):
    t = np.sqrt(x**2+y**2)
    h = np.arctanh(z/np.sqrt(t**2+z**2)) if z!=0. else 0.
    p = BoundPhi(np.arctan2(y, x))
    m2 = e**2 - t**2 - z**2
    m = np.sign(m2) * np.sqrt(abs(m2))
    return t, h, p, m
