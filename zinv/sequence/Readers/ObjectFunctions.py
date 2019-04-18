import numpy as np
import numba as nb
import awkward as awk
import operator
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Geometry import RadToCart2D, CartToRad2D, BoundPhi

@nb.njit(["float32[:](float32[:], float32, float32[:], float32[:])"])
def pt_shift_numba(pt, nsig, up, down):
    return (pt*(1 + (nsig>=0)*nsig*up - (nsig<0)*nsig*down)).astype(np.float32)

def jet_pt_shift(ev, source, nsig):
    nominal = ev.Jet.pt
    try:
        up = getattr(ev.Jet, 'JEC{}Up'.format(source)).content
        down = getattr(ev.Jet, 'JEC{}Down'.format(source)).content
    except AttributeError:
        up = np.zeros_like(nominal.content, dtype=np.float32)
        down = np.zeros_like(nominal.content, dtype=np.float32)
    return awk.JaggedArray(nominal.starts, nominal.stops, pt_shift_numba(
        nominal.content, nsig, up, down,
    ))

def jet_dphimet(ev, source, nsig):
    @nb.njit(["float32[:](float32[:], float32[:], int64[:], int64[:])"])
    def dphi_met(mephi, jphi, starts, stops):
        dphi = np.pi*np.ones_like(jphi, dtype=np.float32)
        for iev, (start, stop) in enumerate(zip(starts, stops)):
            for iob in range(start, stop):
                dphi[iob] = np.abs(BoundPhi(jphi[iob]-mephi[iev]))

        return dphi.astype(np.float32)

    jphi = ev.Jet.phi
    return awk.JaggedArray(
        jphi.starts, jphi.stops, dphi_met(
            ev.MET_phiShift(ev, source, nsig), jphi.content, jphi.starts, jphi.stops,
        ),
    )

def muon_pt_shift(ev, source, nsig):
    shift = (source=="muonPtScale")*ev.Muon.ptErr.content/ev.Muon.pt.content
    return awk.JaggedArray(ev.Muon.pt.starts, ev.Muon.pt.stops, pt_shift_numba(
        ev.Muon.pt.content, nsig, shift, -1.*shift
    ))

def ele_pt_shift(ev, source, nsig):
    shift = (source=="eleEnergyScale")*ev.Electron_energyErr.content/ev.Electron.pt.content
    return awk.JaggedArray(ev.Electron.pt.starts, ev.Electron.pt.stops, pt_shift_numba(
        ev.Electron.pt.content, nsig, shift, -shift
    ))

def photon_pt_shift(ev, source, nsig):
    shift = (source=="photonEnergyScale")*ev.Photon_energyErr.content/ev.Photon.pt.content
    result = awk.JaggedArray(ev.Photon.pt.starts, ev.Photon.pt.stops, pt_shift_numba(
        ev.Photon.pt.content, nsig, shift, -shift
    ))
    result.content[np.isnan(result).content] = 0.
    return result

def met_shift(ev, source, nsig, attr):
    @nb.njit(["UniTuple(float32[:],2)(float32[:],float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],float32[:],float32[:],float32)"])
    def met_shift_numba(
        met, mephi, jpt, jptcorr, jphi, jstarts, jstops, metuncx, metuncy, nsig,
    ):
        jpx_old, jpy_old = RadToCart2D(jpt, jphi)
        jpx_new, jpy_new = RadToCart2D(jptcorr, jphi)

        mex, mey = RadToCart2D(met, mephi)
        for iev, (start, stop) in enumerate(zip(jstarts, jstops)):
            for iob in range(start, stop):
                mex[iev] += (jpx_old[iob] - jpx_new[iob])
                mey[iev] += (jpy_old[iob] - jpy_new[iob])

        mex += nsig*metuncx
        mey += nsig*metuncy

        return CartToRad2D(mex, mey)

    arg_ = 1 if attr=='phi' else 0
    return met_shift_numba(
        ev.MET_pt, ev.MET_phi, ev.Jet_pt.content,
        ev.Jet_ptShift(ev, source, nsig).content, ev.Jet_phi.content,
        ev.Jet_pt.starts, ev.Jet_pt.stops,
        (source=="unclust")*ev.MET_MetUnclustEnUpDeltaX,
        (source=="unclust")*ev.MET_MetUnclustEnUpDeltaY,
        nsig,
    )[arg_].astype(np.float32)

def obj_selection(ev, source, nsig, attr, name, sele, xclean=False):
    mask = getattr(ev, "{}_{}Mask".format(name, sele))(ev, source, nsig)
    if xclean:
        mask = mask & getattr(ev, "{}_XCleanMask".format(name))(ev, source, nsig)

    obj = getattr(ev, "{}_{}".format(name, attr))
    if callable(obj):
        obj = obj(ev)

    return obj[mask]

class ObjectFunctions(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.register_function(event, "Jet_ptShift", jet_pt_shift)
        event.register_function(event, "Muon_ptShift", muon_pt_shift)
        event.register_function(event, "Electron_ptShift", ele_pt_shift)
        event.register_function(event, "Photon_ptShift", photon_pt_shift)
        event.register_function(event, "Tau_ptShift", lambda ev, source, nsig: ev.Tau_pt)
        event.register_function(event, "MET_ptShift", partial(met_shift, attr='pt'))
        event.register_function(event, "MET_phiShift", partial(met_shift, attr='phi'))
        event.register_function(event, "Jet_dphiMET", jet_dphimet)

        for objname, selection, xclean in self.selections:
            print(objname, selection)
            if xclean:
                event.register_function(
                    event, selection+"NoXClean",
                    partial(obj_selection, name=objname, sele=selection),
                )
                event.register_function(
                    event, selection,
                    partial(
                        obj_selection, name=objname, sele=selection,
                        xclean=True,
                    ),
                )
            else:
                event.register_function(
                    event, selection,
                    partial(obj_selection, name=objname, sele=selection),
                )

    def event(self, event):
        print(event.Jet_ptShift(event, '', 0.))
        print(event.Muon_ptShift(event, '', 0.))
        print(event.Electron_ptShift(event, '', 0.))
        print(event.Photon_ptShift(event, '', 0.))
        print(event.Tau_ptShift(event, '', 0.))
        print(event.MET_ptShift(event, '', 0.))
        print(event.MET_phiShift(event, '', 0.))
        print(event.Jet_dphiMET(event, '', 0.))
        print(event.JetSelection(event, '', 0., 'pt'))
