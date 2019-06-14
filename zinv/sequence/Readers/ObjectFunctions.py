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
    updo = 'Up' if nsig>=0. else 'Down'
    if source=='jerSF':
        variation = 1. + np.abs(nsig)*getattr(ev, 'Jet_JECjerSF{}'.format(updo))
    elif source.startswith("jes"):
        variation = 1. + np.abs(nsig)*ev.Jet_jesSF(ev, source, nsig)
    else:
        variation = 1.

    return ev.Jet_pt * variation

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

def tau_dphimet(ev, source, nsig):
    def dphi_met(mephi, tphi, starts, stops):
        dphi = np.pi*np.ones_like(tphi, dtype=np.float32)
        for iev, (start, stop) in enumerate(zip(starts, stops)):
            for iob in range(start, stop):
                dphi[iob] = np.abs(BoundPhi(tphi[iob]-mephi[iev]))

        return dphi.astype(np.float32)

    tphi = ev.Tau.phi
    return awk.JaggedArray(
        tphi.starts, tphi.stops, dphi_met(
            ev.MET_phiShift(ev, source, nsig), tphi.content, tphi.starts, tphi.stops,
        ),
    )

def muon_pt_shift(ev, source, nsig):
    shift = ((source=="muonPtScale")*ev.Muon.ptErr.content/ev.Muon.pt.content).astype(np.float32)
    return awk.JaggedArray(ev.Muon.pt.starts, ev.Muon.pt.stops, pt_shift_numba(
        ev.Muon.pt.content, nsig, shift, -1.*shift
    ))

def muon_pt_met(ev, source, nsig):
    shift = ((source=="muonPtScale")*0.002*np.ones_like(ev.Muon.pt.content)).astype(np.float32)
    return awk.JaggedArray(ev.Muon.pt.starts, ev.Muon.pt.stops, pt_shift_numba(
        ev.Muon.pt.content, nsig, shift, -1.*shift
    ))

def ele_pt_shift(ev, source, nsig):
    shift = ((source=="eleEnergyScale")*ev.Electron_energyErr.content/ev.Electron.pt.content).astype(np.float32)
    return awk.JaggedArray(ev.Electron.pt.starts, ev.Electron.pt.stops, pt_shift_numba(
        ev.Electron.pt.content, nsig, shift, -shift
    ))

def ele_pt_met(ev, source, nsig):
    shift = ((source=="eleEnergyScale")*(0.006*(ev.Electron_eta<=1.479) + 0.015*(ev.Electron_eta>1.479))).astype(np.float32).content
    return awk.JaggedArray(ev.Electron.pt.starts, ev.Electron.pt.stops, pt_shift_numba(
        ev.Electron.pt.content, nsig, shift, -shift
    ))

def photon_pt_shift(ev, source, nsig):
    shift = ((source=="photonEnergyScale")*ev.Photon_energyErr.content/ev.Photon.pt.content).astype(np.float32)
    result = awk.JaggedArray(ev.Photon.pt.starts, ev.Photon.pt.stops, pt_shift_numba(
        ev.Photon.pt.content, nsig, shift, -shift
    ))
    result.content[np.isnan(result).content] = 0.
    return result

def photon_pt_met(ev, source, nsig):
    shift = ((source=="photonEnergyScale")*(0.01*(ev.Photon_eta<=1.479) + 0.025*(ev.Photon_eta>1.479))).astype(np.float32).content
    result = awk.JaggedArray(ev.Photon.pt.starts, ev.Photon.pt.stops, pt_shift_numba(
        ev.Photon.pt.content, nsig, shift, -shift
    ))
    result.content[np.isnan(result).content] = 0.
    return result

def met_shift(ev, source, nsig, attr):
    @nb.njit([
        "UniTuple(float32[:],2)("
        "float32[:],float32[:],"
        "float32[:],float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],float32"
        ")"
    ])
    def met_shift_numba(
        met, mephi,
        jpt, jptcorr, jphi, jstarts, jstops,
        ept, eptcorr, ephi, estarts, estops,
        mpt, mptcorr, mphi, mstarts, mstops,
        ppt, pptcorr, pphi, pstarts, pstops,
        tpt, tptcorr, tphi, tstarts, tstops,
        metuncx, metuncy, nsig,
    ):
        jpx_old, jpy_old = RadToCart2D(jpt, jphi)
        jpx_new, jpy_new = RadToCart2D(jptcorr, jphi)
        epx_old, epy_old = RadToCart2D(ept, ephi)
        epx_new, epy_new = RadToCart2D(eptcorr, ephi)
        mpx_old, mpy_old = RadToCart2D(mpt, mphi)
        mpx_new, mpy_new = RadToCart2D(mptcorr, mphi)
        ppx_old, ppy_old = RadToCart2D(ppt, pphi)
        ppx_new, ppy_new = RadToCart2D(pptcorr, pphi)
        tpx_old, tpy_old = RadToCart2D(tpt, tphi)
        tpx_new, tpy_new = RadToCart2D(tptcorr, tphi)

        mex, mey = RadToCart2D(met, mephi)
        for iev, (jsta, jsto, esta, esto, msta, msto, psta, psto, tsta, tsto) in enumerate(zip(
            jstarts, jstops, estarts, estops, mstarts, mstops, pstarts, pstops, tstarts, tstops,
        )):
            for iob in range(jsta, jsto):
                mex[iev] += (jpx_old[iob] - jpx_new[iob])
                mey[iev] += (jpy_old[iob] - jpy_new[iob])
            for iob in range(esta, esto):
                mex[iev] += (epx_old[iob] - epx_new[iob])
                mey[iev] += (epy_old[iob] - epy_new[iob])
            for iob in range(msta, msto):
                mex[iev] += (mpx_old[iob] - mpx_new[iob])
                mey[iev] += (mpy_old[iob] - mpy_new[iob])
            for iob in range(psta, psto):
                mex[iev] += (ppx_old[iob] - ppx_new[iob])
                mey[iev] += (ppy_old[iob] - ppy_new[iob])
            for iob in range(tsta, tsto):
                mex[iev] += (tpx_old[iob] - tpx_new[iob])
                mey[iev] += (tpy_old[iob] - tpy_new[iob])

        mex += nsig*metuncx
        mey += nsig*metuncy

        return CartToRad2D(mex, mey)

    arg_ = 1 if attr=='phi' else 0
    photon_mask = (~np.isnan(ev.Photon_pt))
    return met_shift_numba(
        ev.MET_pt, ev.MET_phi,
        ev.Jet_pt.content, ev.Jet_ptShift(ev, source, nsig).content,
        ev.Jet_phi.content, ev.Jet_pt.starts, ev.Jet_pt.stops,
        ev.ElectronSelection(ev, source, nsig, 'pt').content,
        ev.ElectronSelection(ev, source, nsig, 'ptShift').content,
        ev.ElectronSelection(ev, source, nsig, 'phi').content,
        ev.ElectronSelection(ev, source, nsig, 'phi').starts,
        ev.ElectronSelection(ev, source, nsig, 'phi').stops,
        ev.MuonSelection(ev, source, nsig, 'pt').content,
        ev.MuonSelection(ev, source, nsig, 'ptShift').content,
        ev.MuonSelection(ev, source, nsig, 'phi').content,
        ev.MuonSelection(ev, source, nsig, 'phi').starts,
        ev.MuonSelection(ev, source, nsig, 'phi').stops,
        ev.PhotonSelection(ev, source, nsig, 'pt').content,
        ev.PhotonSelection(ev, source, nsig, 'ptShift').content,
        ev.PhotonSelection(ev, source, nsig, 'phi').content,
        ev.PhotonSelection(ev, source, nsig, 'phi').starts,
        ev.PhotonSelection(ev, source, nsig, 'phi').stops,
        ev.TauSelection(ev, source, nsig, 'pt').content,
        ev.TauSelection(ev, source, nsig, 'ptShift').content,
        ev.TauSelection(ev, source, nsig, 'phi').content,
        ev.TauSelection(ev, source, nsig, 'phi').starts,
        ev.TauSelection(ev, source, nsig, 'phi').stops,
        (source=="unclust")*ev.MET_MetUnclustEnUpDeltaX,
        (source=="unclust")*ev.MET_MetUnclustEnUpDeltaY,
        nsig,
    )[arg_].astype(np.float32)

def met_sumet_shift(ev, source, nsig):
    @nb.njit(["float32[:]("
        "float32[:],"
        "float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],int64[:],int64[:],"
        "float32[:],float32[:],int64[:],int64[:])"
    ])
    def nb_met_sumet_shift(
        sumet,
        jpt, cjpt, jstas, jstos,
        ept, cept, estas, estos,
        mpt, cmpt, mstas, mstos,
        ypt, cypt, ystas, ystos,
        tpt, ctpt, tstas, tstos,
    ):
        sumet_shift = np.zeros_like(sumet, dtype=np.float32)

        for iev, (jsta, jsto, esta, esto, msta, msto, ysta, ysto, tsta, tsto) in enumerate(zip(
            jstas, jstos, estas, estos, mstas, mstos, ystas, ystos, tstas, tstos,
        )):
            sumet_shift[iev] = (
                sumet[iev]
                + (cjpt[jsta:jsto] - jpt[jsta:jsto]).sum()
                + (cept[esta:esto] - ept[esta:esto]).sum()
                + (cmpt[msta:msto] - mpt[msta:msto]).sum()
                + (cypt[ysta:ysto] - ypt[ysta:ysto]).sum()
                + (ctpt[tsta:tsto] - tpt[tsta:tsto]).sum()
            )
        return sumet_shift

    return nb_met_sumet_shift(
        ev.MET_sumEt,
        ev.JetSelection(ev, source, nsig, 'pt').content,
        ev.JetSelection(ev, source, nsig, 'ptShift').content,
        ev.JetSelection(ev, source, nsig, 'eta').starts,
        ev.JetSelection(ev, source, nsig, 'eta').stops,
        ev.ElectronSelection(ev, source, nsig, 'pt').content,
        ev.ElectronSelection(ev, source, nsig, 'ptShift').content,
        ev.ElectronSelection(ev, source, nsig, 'eta').starts,
        ev.ElectronSelection(ev, source, nsig, 'eta').stops,
        ev.MuonSelection(ev, source, nsig, 'pt').content,
        ev.MuonSelection(ev, source, nsig, 'ptShift').content,
        ev.MuonSelection(ev, source, nsig, 'eta').starts,
        ev.MuonSelection(ev, source, nsig, 'eta').stops,
        ev.PhotonSelection(ev, source, nsig, 'pt').content,
        ev.PhotonSelection(ev, source, nsig, 'ptShift').content,
        ev.PhotonSelection(ev, source, nsig, 'eta').starts,
        ev.PhotonSelection(ev, source, nsig, 'eta').stops,
        ev.TauSelection(ev, source, nsig, 'pt').content,
        ev.TauSelection(ev, source, nsig, 'ptShift').content,
        ev.TauSelection(ev, source, nsig, 'eta').starts,
        ev.TauSelection(ev, source, nsig, 'eta').stops,
    )

def obj_selection(ev, source, nsig, attr, name, sele, xclean=False):
    mask = getattr(ev, "{}_{}Mask".format(name, sele))(ev, source, nsig)
    if xclean:
        mask = mask & getattr(ev, "{}_XCleanMask".format(name))(ev, source, nsig)

    obj = getattr(ev, "{}_{}".format(name, attr))
    if callable(obj):
        obj = obj(ev, source, nsig)

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
        event.register_function(event, "MET_sumEtShift", partial(met_sumet_shift))
        event.register_function(event, "Jet_dphiMET", jet_dphimet)
        event.register_function(event, "Tau_dphiMET", tau_dphimet)

        for objname, selection, xclean in self.selections:
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
