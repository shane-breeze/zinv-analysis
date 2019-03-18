import numpy as np
import numba as nb
import awkward as awk
import operator
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from utils.Geometry import RadToCart2D, CartToRad2D, BoundPhi

@nb.njit(["float32[:](float32[:], float32, float32[:], float32[:])"])
def pt_shift_numba(pt, nsig, up, down):
    return (pt*(1 + (nsig>=0)*nsig*up - (nsig<0)*nsig*down)).astype(np.float32)

def jet_pt_shift():
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fjet_pt_shift'))
    def fjet_pt_shift(ev, evidx, nsig, source):
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

    def return_jet_pt_shift(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fjet_pt_shift(ev, ev.iblock, nsig, source)

    return return_jet_pt_shift

def jet_dphimet():
    @nb.njit(["float32[:](float32[:], float32[:], int64[:], int64[:])"])
    def dphi_met(mephi, jphi, starts, stops):
        dphi = np.pi*np.ones_like(jphi, dtype=np.float32)
        for iev, (start, stop) in enumerate(zip(starts, stops)):
            for iob in range(start, stop):
                dphi[iob] = np.abs(BoundPhi(jphi[iob]-mephi[iev]))

        return dphi.astype(np.float32)

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fjet_dphimet'))
    def fjet_dphimet(ev, evidx, nsig, source):
        jphi = ev.Jet.phi
        return dphi_met(
            ev.MET_phiShift(ev), jphi.content, jphi.starts, jphi.stops,
        )

    def return_jet_dphimet(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fjet_dphimet(ev, ev.iblock, nsig, source)

    return return_jet_dphimet

def muon_pt_shift():
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fmuon_pt_shift'))
    def fmuon_pt_shift(ev, evidx, nsig, source):
        shift = (source=="muonPtScale")*ev.Muon.ptErr.content/ev.Muon.pt.content
        return awk.JaggedArray(ev.Muon.pt.starts, ev.Muon.pt.stops, pt_shift_numba(
            ev.Muon.pt.content, nsig, shift, -1.*shift
        ))

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ['muonPtScale']:
            source, nsig = '', 0.
        return fmuon_pt_shift(ev, ev.iblock, nsig, source)

    return ret_func

def ele_pt_shift():
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fele_pt_shift'))
    def fele_pt_shift(ev, evidx, nsig, source):
        shift = (source=="eleEnergyScale")*ev.Electron_energyErr.content/ev.Electron.pt.content
        return awk.JaggedArray(ev.Electron.pt.starts, ev.Electron.pt.stops, pt_shift_numba(
            ev.Electron.pt.content, nsig, shift, -shift
        ))

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ['eleEnergyScale']:
            source, nsig = '', 0.
        return fele_pt_shift(ev, ev.iblock, nsig, source)

    return ret_func

def photon_pt_shift():
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fphoton_pt_shift'))
    def fphoton_pt_shift(ev, evidx, nsig, source):
        shift = (source=="photonEnergyScale")*ev.Photon_energyErr.content/ev.Photon.pt.content
        result = awk.JaggedArray(ev.Photon.pt.starts, ev.Photon.pt.stops, pt_shift_numba(
            ev.Photon.pt.content, nsig, shift, -shift
        ))
        result.content[np.isnan(result).content] = 0.
        return result

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ['photonEnergyScale']:
            source, nsig = '', 0.
        return fphoton_pt_shift(ev, ev.iblock, nsig, source)

    return ret_func

def met_shift(arg):
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

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fmet_shift'))
    def fmet_shift(ev, evidx, nsig, source, arg_):
        return met_shift_numba(
            ev.MET_pt, ev.MET_phi, ev.Jet_pt.content,
            ev.Jet_ptShift(ev).content, ev.Jet_phi.content,
            ev.Jet_pt.starts, ev.Jet_pt.stops,
            (source=="unclust")*ev.MET_MetUnclustEnUpDeltaX,
            (source=="unclust")*ev.MET_MetUnclustEnUpDeltaY,
            nsig,
        )[arg_].astype(np.float32)

    def return_met_shift(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fmet_shift(ev, ev.iblock, nsig, source, arg)

    return return_met_shift

def obj_selection(objname, selection, xclean=False):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fobj_selection'))
    def fobj_selection(ev, evidx, nsig, source, objname_, selection_, xclean_, attr):
        mask = getattr(ev, "{}_{}Mask".format(objname_, selection_))(ev)
        if xclean_:
            mask = mask & getattr(ev, "{}_XCleanMask".format(objname_))(ev)

        obj = getattr(ev, "{}_{}".format(objname_, attr))
        if callable(obj):
            obj = obj(ev)

        return obj[mask]

    def return_obj_selection(ev, attr):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fobj_selection(
            ev, ev.iblock, nsig, source, objname, selection, xclean, attr,
        )

    return return_obj_selection

class ObjectFunctions(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.Jet_ptShift = jet_pt_shift()
        event.Muon_ptShift = muon_pt_shift()
        event.Electron_ptShift = ele_pt_shift()
        event.Photon_ptShift = photon_pt_shift()
        event.Tau_ptShift = lambda ev: ev.Tau_pt
        event.MET_ptShift = met_shift(0)
        event.MET_phiShift = met_shift(1)
        event.Jet_dphiMET = jet_dphimet()

        for objname, selection, xclean in self.selections:
            if xclean:
                setattr(event, selection+"NoXClean", obj_selection(objname, selection))
                setattr(event, selection, obj_selection(objname, selection, xclean=True))
            else:
                setattr(event, selection, obj_selection(objname, selection))
