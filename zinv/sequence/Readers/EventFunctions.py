import numpy as np
import numba as nb
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Geometry import (BoundPhi, RadToCart2D, CartToRad2D,
                                 LorTHPMToXYZE, LorXYZEToTHPM)

def evaluate_metnox(arg):
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

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_metnox'))
    def fevaluate_metnox(ev, evidx, nsig, source, arg_):
        return metnox_numba(
            ev.MET_ptShift(ev, source, nsig), ev.MET_phiShift(ev, source, nsig),
            ev.MuonSelection(ev, source, nsig, 'ptShift').content, ev.MuonSelection(ev, source, nsig, 'phi').content,
            ev.MuonSelection(ev, source, nsig, 'phi').starts, ev.MuonSelection(ev, source, nsig, 'phi').stops,
            ev.ElectronSelection(ev, source, nsig, 'ptShift').content, ev.ElectronSelection(ev, source, nsig, 'phi').content,
            ev.ElectronSelection(ev, source, nsig, 'phi').starts, ev.ElectronSelection(ev, source, nsig, 'phi').stops,
        )[arg_].astype(np.float32)

    def return_evaluate_metnox(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        if source in ["muonPtScale", "eleEnergyScale"]:
            source, nsig = '', 0.
        return fevaluate_metnox(ev, ev.iblock, nsig, source, arg)

    return return_evaluate_metnox

def evaluate_mindphi(njets):
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

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_mindphi'))
    def fevaluate_mindphi(ev, evidx, nsig, source, njets_):
        jphis = ev.JetSelection(ev, 'phi')
        return mindphi_numba(
            jphis.content, jphis.starts, jphis.stops, njets_, ev.METnoX_phi(ev),
        )

    def return_evaluate_mindphi(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_mindphi(ev, ev.iblock, nsig, source, njets)

    return return_evaluate_mindphi

def evaluate_met_dcalo():
    @nb.njit(["float32[:](float32[:],float32[:],float32[:])"])
    def met_dcalo_numba(pfmet, calomet, metnox):
        return np.abs(pfmet-calomet)/metnox

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_met_dcalo'))
    def fevaluate_met_dcalo(ev, evidx, nsig, source):
        return met_dcalo_numba(
            ev.MET_ptShift(ev, source, nsig), ev.CaloMET_pt, ev.METnoX_pt(ev, source, nsig),
        )

    def return_evaluate_met_dcalo(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_met_dcalo(ev, ev.iblock, nsig, source)

    return return_evaluate_met_dcalo

def evaluate_mtw():
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

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_mtw'))
    def fevaluate_mtw(ev, evidx, nsig, source):
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

def evaluate_mll():
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

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_mll'))
    def fevaluate_mll(ev, evidx, nsig, source):
        mpts = ev.MuonSelection(ev, source, nsig, 'ptShift')
        epts = ev.ElectronSelection(ev, source, nsig, 'ptShift')
        return event_mll_numba(
            mpts.content, ev.MuonSelection(ev, source, nsig, 'eta').content,
            ev.MuonSelection(ev, source, nsig, 'phi').content, ev.MuonSelection(ev, source, nsig, 'mass').content,
            mpts.starts, mpts.stops,
            epts.content, ev.ElectronSelection(ev, source, nsig, 'eta').content,
            ev.ElectronSelection(ev, source, nsig, 'phi').content, ev.ElectronSelection(ev, source, nsig, 'mass').content,
            epts.starts, epts.stops,
            ev.size,
        )

    def return_evaluate_mll(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_mll(ev, ev.iblock, nsig, source)

    return return_evaluate_mll

def evaluate_lepton_charge():
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

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_lepton_charge'))
    def fevaluate_lepton_charge(ev, evidx, nsig, source):
        mcharge = ev.MuonSelection(ev, source, nsig, 'charge')
        echarge = ev.ElectronSelection(ev, source, nsig, 'charge')
        return lepton_charge_numba(
            mcharge.content, mcharge.starts, mcharge.stops,
            echarge.content, echarge.starts, echarge.stops,
            ev.size,
        )

    def return_evaluate_lepton_charge(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_lepton_charge(ev, ev.iblock, nsig, source)

    return return_evaluate_lepton_charge

class EventFunctions(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.METnoX_pt = evaluate_metnox(0)
        event.METnoX_phi = evaluate_metnox(1)
        event.MinDPhiJ1234METnoX = evaluate_mindphi(4)
        event.MET_dCaloMET = evaluate_met_dcalo()
        event.MTW = evaluate_mtw()
        event.MLL = evaluate_mll()
        event.LeptonCharge = evaluate_lepton_charge()
