import numpy as np
import awkward as awk
from numba import njit, float32
from cachetools.func import lru_cache

from utils.Geometry import RadToCart2D, CartToRad2D

@njit
def pt_shift_numba(pt, nsig, up, down):
    return pt*(1 + (nsig>=0)*nsig*up - (nsig<0)*nsig*down)

def jet_pt_shift():
    @lru_cache(maxsize=32)
    def fjet_pt_shift(ev, evidx, nsig, source):
        nominal = ev.Jet.pt
        try:
            up = getattr(ev, 'JEC{}Up'.format(source))(ev).content
            down = getattr(ev, 'JEC{}Down'.format(source))(ev).content
        except AttributeError:
            up = 0.
            down = 0.
        return awk.JaggedArray(nominal.starts, nominal.stops, pt_shift_numba(
            nominal.content, nsig, up, down,
        ))
    return lambda ev: fjet_pt_shift(ev, ev.iblock, ev.nsig, ev.source)

def muon_pt_shift():
    @lru_cache(maxsize=32)
    def fmuon_pt_shift(ev, evidx, nsig, source):
        shift = (source=="muonPtScale")*ev.Muon_ptErr.content/ev.Muon.pt.content
        return awk.JaggedArray(ev.Muon.pt.starts, ev.Muon.pt.stops, pt_shift_numba(
            ev.Muon.pt.content, nsig, shift, -1.*shift
        ))
    return lambda ev: fmuon_pt_shift(ev, ev.iblock, ev.nsig, ev.source)

def ele_pt_shift():
    @lru_cache(maxsize=32)
    def fele_pt_shift(ev, evidx, nsig, source):
        shift = (source=="eleEnergyScale")*ev.Electron_energyErr.content/ev.Electron.pt.content
        return awk.JaggedArray(ev.Electron.pt.starts, ev.Electron.pt.stops, pt_shift_numba(
            ev.Electron.pt.content, nsig, shift, -shift
        ))
    return lambda ev: fele_pt_shift(ev, ev.iblock, ev.nsig, ev.source)

def photon_pt_shift():
    @lru_cache(maxsize=32)
    def fphoton_pt_shift(ev, evidx, nsig, source):
        shift = (source=="photonEnergyScale")*ev.Photon_energyErr.content/ev.Photon.pt.content
        result = awk.JaggedArray(ev.Photon.pt.starts, ev.Photon.pt.stops, pt_shift_numba(
            ev.Photon.pt.content, nsig, shift, -shift
        ))
        result.content[np.isnan(result).content] = 0.
        return result
    return lambda ev: fphoton_pt_shift(ev, ev.iblock, ev.nsig, ev.source)

def met_shift(arg):
    @njit
    def met_shift_numba(
        met, mephi, jpt, jptcorr, jphi, jstarts, jstops, metuncx, metuncy, nsig,
    ):
        jpx_old, jpy_old = RadToCart2D(jpt, jphi)
        jpx_new, jpy_new = RadToCart2D(jptcorr, jphi)
        djpx = jpx_new - jpx_old
        djpy = jpy_new - jpy_old

        mex, mey = RadToCart2D(met, mephi)
        for idx, (start, stop) in enumerate(zip(jstarts, jstops)):
            mex[idx] -= djpx[start:stop][jpx_new[start:stop]>15.].sum()
            mey[idx] -= djpy[start:stop][jpy_new[start:stop]>15.].sum()
        mex += nsig*metuncx
        mey += nsig*metuncy

        return CartToRad2D(mex, mey)

    @lru_cache(maxsize=32)
    def fmet_shift(ev, evidx, nsig, source, arg_):
        return met_shift_numba(
            ev.MET_pt, ev.MET_phi, ev.Jet_pt.content,
            ev.Jet_ptShift(ev).content, ev.Jet_phi.content,
            ev.Jet_pt.starts, ev.Jet_pt.stops,
            (source=="unclust")*ev.MET_MetUnclustEnUpDeltaX,
            (source=="unclust")*ev.MET_MetUnclustEnUpDeltaY,
            nsig,
        )[arg_]
    def fmet_shift_no_source(ev, arg_):
        return fmet_shift(ev, ev.iblock, ev.nsig, ev.source, arg_)
    return lambda ev: fmet_shift_no_source(ev, arg)

def obj_selection(objname, selection, xclean=False):
    @lru_cache(maxsize=32)
    def fobj_selection(ev, evidx, nsig, source, objname_, selection_, xclean_, attr):
        mask = getattr(ev, "{}_{}Mask".format(objname_, selection_))(ev)
        if xclean_:
            mask = mask & getattr(ev, "{}_XCleanMask".format(objname_))(ev)

        obj = getattr(ev, "{}_{}".format(objname_, attr))
        if callable(obj):
            obj = obj(ev)

        return obj[mask]

    return lambda ev, attr: fobj_selection(
        ev, ev.iblock, ev.nsig, ev.source, objname, selection, xclean, attr,
    )

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

        for objname, selection, xclean in self.selections:
            if xclean:
                setattr(event, selection+"NoXClean", obj_selection(objname, selection))
                setattr(event, selection, obj_selection(objname, selection, xclean=True))
            else:
                setattr(event, selection, obj_selection(objname, selection))
