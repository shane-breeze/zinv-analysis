import numpy as np
import numba as nb
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Geometry import (
    BoundPhi, RadToCart2D, CartToRad2D, LorTHPMToXYZE, LorXYZEToTHPM
)

def evaluate_metnox(ev, source, nsig, attr):
    @nb.njit(["UniTuple(float32[:],2)(float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],float32[:],float32[:],int64[:],int64[:])"])
    def metnox_numba(
        met, mephi,
        mupt, muphi, mustarts, mustops,
        elept, elephi, elestarts, elestops,
    ):
        mex, mey = RadToCart2D(met, mephi)
        mux, muy = RadToCart2D(mupt, muphi)
        elex, eley = RadToCart2D(elept, elephi)

        for idx, (mstart, mstop, estart, estop) in enumerate(zip(
            mustarts, mustops, elestarts, elestops
        )):
            mex[idx] += mux[mstart:mstop].sum() + elex[estart:estop].sum()
            mey[idx] += muy[mstart:mstop].sum() + eley[estart:estop].sum()

        return CartToRad2D(mex, mey)

    arg = 1 if attr=='phi' else 0
    return metnox_numba(
        ev.MET_ptShift(ev, source, nsig), ev.MET_phiShift(ev, source, nsig),
        ev.MuonSelection(ev, source, nsig, 'ptShift').content,
        ev.MuonSelection(ev, source, nsig, 'phi').content,
        ev.MuonSelection(ev, source, nsig, 'phi').starts,
        ev.MuonSelection(ev, source, nsig, 'phi').stops,
        ev.ElectronSelection(ev, source, nsig, 'ptShift').content,
        ev.ElectronSelection(ev, source, nsig, 'phi').content,
        ev.ElectronSelection(ev, source, nsig, 'phi').starts,
        ev.ElectronSelection(ev, source, nsig, 'phi').stops,
    )[arg].astype(np.float32)

def evaluate_metnox_sumet(ev, source, nsig):
    @nb.njit(["float32[:](float32[:], float32[:], int64[:], int64[:], float32[:], int64[:], int64[:])"])
    def nb_evaluate_metnox_sumet(
        sumet, mpt, mstas, mstos, ept, estas, estos,
    ):
        csumet = np.zeros_like(sumet, dtype=np.float32)

        for iev, (msta, msto, esta, esto) in enumerate(zip(
            mstas, mstos, estas, estos,
        )):
            csumet[iev] = sumet[iev] - mpt[msta:msto].sum() - ept[esta:esto].sum()

        return csumet

    return nb_evaluate_metnox_sumet(
        ev.MET_sumEtShift(ev, source, nsig),
        ev.MuonSelection(ev, source, nsig, 'ptShift').content,
        ev.MuonSelection(ev, source, nsig, 'phi').starts,
        ev.MuonSelection(ev, source, nsig, 'phi').stops,
        ev.ElectronSelection(ev, source, nsig, 'ptShift').content,
        ev.ElectronSelection(ev, source, nsig, 'phi').starts,
        ev.ElectronSelection(ev, source, nsig, 'phi').stops,
    )

def evaluate_mindphi(ev, source, nsig, njets):
    @nb.njit(["float32[:](float32[:],int64[:],int64[:],int64,float32[:])"])
    def mindphi_numba(jphi, jstarts, jstops, njets_, mephi):
        dphi = np.zeros_like(mephi, dtype=np.float32)
        for iev, (start, stops) in enumerate(zip(jstarts, jstops)):
            delta = min(njets_, stops-start)
            if delta>0:
                dphi[iev] = np.abs(BoundPhi(
                    jphi[start:start+delta] - mephi[iev]
                )).min()
            else:
                dphi[iev] = np.nan
        return dphi

    jphis = ev.JetSelection(ev, source, nsig, 'phi')
    return mindphi_numba(
        jphis.content, jphis.starts, jphis.stops, njets,
        ev.METnoX_phi(ev, source, nsig),
    )

def evaluate_met_dcalo(ev, source, nsig):
    @nb.njit(["float32[:](float32[:],float32[:],float32[:])"])
    def met_dcalo_numba(pfmet, calomet, metnox):
        return np.abs(pfmet-calomet)/metnox

    return met_dcalo_numba(
        ev.MET_ptShift(ev, source, nsig), ev.CaloMET_pt,
        ev.METnoX_pt(ev, source, nsig),
    )

    def return_evaluate_met_dcalo(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_met_dcalo(ev, ev.iblock, nsig, source)

    return return_evaluate_met_dcalo

def evaluate_mtw(ev, source, nsig):
    @nb.njit(["float32(float32,float32)"])
    def mtw_numba(ptprod, dphi):
        return np.sqrt(2*ptprod*(1-np.cos(dphi)))

    @nb.njit(["float32[:](float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],float32[:],float32[:],int64[:],int64[:])"])
    def event_mtw_numba(
        met, mephi,
        mupt, muphi, mustarts, mustops,
        elept, elephi, elestarts, elestops,
    ):
        mtw = np.zeros_like(met, dtype=np.float32)
        for iev, (msta, msto, esta, esto) in enumerate(zip(
            mustarts, mustops, elestarts, elestops
        )):
            if msto-msta == 1 and esto-esta == 0:
                mtw[iev] = mtw_numba(
                    met[iev]*mupt[msta], mephi[iev]-muphi[msta],
                )
            elif esto-esta == 1 and msto-msta == 0:
                mtw[iev] = mtw_numba(
                    met[iev]*elept[esta], mephi[iev]-elephi[esta],
                )
            else:
                mtw[iev] = np.nan

        return mtw

    mupts = ev.MuonSelection(ev, source, nsig, 'ptShift')
    epts = ev.ElectronSelection(ev, source, nsig, 'ptShift')
    return event_mtw_numba(
        ev.MET_ptShift(ev, source, nsig), ev.MET_phiShift(ev, source, nsig),
        mupts.content, ev.MuonSelection(ev, source, nsig, 'phi').content,
        mupts.starts, mupts.stops,
        epts.content, ev.ElectronSelection(ev, source, nsig, 'phi').content,
        epts.starts, epts.stops,
    )

    def return_evaluate_mtw(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_mtw(ev, ev.iblock, nsig, source)

    return return_evaluate_mtw

def evaluate_mll(ev, source, nsig):
    @nb.njit(["float32[:](float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],int64)"])
    def event_mll_numba(
        mpt, meta, mphi, mmass, mstas, mstos,
        ept, eeta, ephi, emass, estas, estos,
        evsize,
    ):
        mll = np.zeros(evsize, dtype=np.float32)
        for iev, (msta, msto, esta, esto) in enumerate(zip(
            mstas, mstos, estas, estos,
        )):
            if msto - msta == 2 and esto - esta == 0:
                x1, y1, z1, e1 = LorTHPMToXYZE(
                    mpt[msta], meta[msta], mphi[msta], mmass[msta],
                )
                x2, y2, z2, e2 = LorTHPMToXYZE(
                    mpt[msta+1], meta[msta+1], mphi[msta+1], mmass[msta+1],
                )
                mll[iev] = LorXYZEToTHPM(x1+x2, y1+y2, z1+z2, e1+e2)[-1]
            elif esto - esta == 2 and msto - msta == 0:
                x1, y1, z1, e1 = LorTHPMToXYZE(
                    ept[esta], eeta[esta], ephi[esta], emass[esta],
                )
                x2, y2, z2, e2 = LorTHPMToXYZE(
                    ept[esta+1], eeta[esta+1], ephi[esta+1], emass[esta+1],
                )
                mll[iev] = LorXYZEToTHPM(x1+x2, y1+y2, z1+z2, e1+e2)[-1]
            else:
                mll[iev] = np.nan

        return mll

    mpts = ev.MuonSelection(ev, source, nsig, 'ptShift')
    epts = ev.ElectronSelection(ev, source, nsig, 'ptShift')
    return event_mll_numba(
        mpts.content, ev.MuonSelection(ev, source, nsig, 'eta').content,
        ev.MuonSelection(ev, source, nsig, 'phi').content,
        ev.MuonSelection(ev, source, nsig, 'mass').content,
        mpts.starts, mpts.stops,
        epts.content, ev.ElectronSelection(ev, source, nsig, 'eta').content,
        ev.ElectronSelection(ev, source, nsig, 'phi').content,
        ev.ElectronSelection(ev, source, nsig, 'mass').content,
        epts.starts, epts.stops,
        ev.size,
    )

    def return_evaluate_mll(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_mll(ev, ev.iblock, nsig, source)

    return return_evaluate_mll

def evaluate_lepton_charge(ev, source, nsig):
    @nb.njit(["int32[:](int32[:],int64[:],int64[:],int32[:],int64[:],int64[:],int64)"])
    def lepton_charge_numba(mcharge, mstas, mstos, echarge, estas, estos, evsize):
        charge = np.zeros(evsize, dtype=np.int32)
        for iev, (msta, msto, esta, esto) in enumerate(zip(
            mstas, mstos, estas, estos,
        )):
            if msto - msta > 0:
                charge[iev] += mcharge[msta:msto].sum()
            if esto - esta > 0:
                charge[iev] += echarge[esta:esto].sum()

        return charge

    mcharge = ev.MuonSelection(ev, source, nsig, 'charge')
    echarge = ev.ElectronSelection(ev, source, nsig, 'charge')
    return lepton_charge_numba(
        mcharge.content, mcharge.starts, mcharge.stops,
        echarge.content, echarge.starts, echarge.stops,
        ev.size,
    )

class EventFunctions(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.register_function(event, "METnoX_pt", partial(evaluate_metnox, attr='pt'))
        event.register_function(event, "METnoX_phi", partial(evaluate_metnox, attr='phi'))
        event.register_function(event, "METnoX_sumEt", partial(evaluate_metnox_sumet))
        event.register_function(event, "MinDPhiJ1234METnoX", partial(evaluate_mindphi, njets=4))
        event.register_function(event, "MET_dCaloMET", evaluate_met_dcalo)
        event.register_function(event, "MTW", evaluate_mtw)
        event.register_function(event, "MLL", evaluate_mll)
        event.register_function(event, "LeptonCharge", evaluate_lepton_charge)
