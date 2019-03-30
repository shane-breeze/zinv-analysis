import numpy as np
import numba as nb

@nb.vectorize(["float32(float32)","float64(float64)"])
def BoundPhi(phi):
    if phi >= np.pi:
        phi -= 2*np.pi
    elif phi < -np.pi:
        phi += 2*np.pi
    return phi

@nb.njit([
    "float32(float32,float32)",
    "float32[:](float32[:],float32[:])",
])
def DeltaR2(deta, dphi):
    return deta**2 + BoundPhi(dphi)**2

@nb.njit(["UniTuple(float32[:],2)(float32[:],float32[:])"])
def RadToCart2D(r, phi):
    return r*np.cos(phi), r*np.sin(phi)

@nb.njit(["UniTuple(float32[:],2)(float32[:],float32[:])"])
def CartToRad2D(x, y):
    return np.sqrt(x**2+y**2), BoundPhi(np.arctan2(y, x))

@nb.njit(["UniTuple(float32[:],3)(float32[:],float32[:],float32[:])"])
def PartCoorToCart3D(pt, eta, phi):
    x, y = RadToCart2D(pt, phi)
    z = pt*np.sinh(eta)
    return x, y, z

@nb.njit(["UniTuple(float32[:],3)(float32[:],float32[:],float32[:])"])
def CartToPartCoor3D(x, y, z):
    pt, phi = CartToRad2D(x, y)
    eta = np.arctanh(z/np.sqrt(z**2+pt**2))
    return pt, eta, phi

@nb.njit([
    "UniTuple(float32,4)(float32,float32,float32,float32)",
    "UniTuple(float32[:],4)(float32[:],float32[:],float32[:],float32[:])",
])
def LorTHPMToXYZE(t, h, p, m):
    x = t*np.cos(p)
    y = t*np.sin(p)
    z = t*np.sinh(h)
    e = np.sqrt(m**2 + t**2 + z**2)
    return x, y, z, e

@nb.njit(["UniTuple(float32,4)(float32,float32,float32,float32)"])
def LorXYZEToTHPM(x, y, z, e):
    t = np.sqrt(x**2+y**2)
    h = np.arctanh(z/np.sqrt(t**2+z**2)) if z!=0. else 0.
    p = BoundPhi(np.arctan2(y, x))
    m2 = e**2 - t**2 - z**2
    m = np.sign(m2) * np.sqrt(abs(m2))
    return t, h, p, m
