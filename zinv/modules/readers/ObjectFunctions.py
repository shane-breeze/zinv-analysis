import numpy as np
import numba as nb
import awkward as awk
import operator
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Geometry import RadToCart2D, CartToRad2D, BoundPhi, DeltaR2
from zinv.utils.Lambda import Lambda

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

def jet_dphimet(ev, source, nsig, coll):
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
            getattr(ev, "{}_phiShift".format(coll))(ev, source, nsig),
            jphi.content, jphi.starts, jphi.stops,
        ),
    )

def tau_dphimet(ev, source, nsig, coll):
    def dphi_met(mephi, tphi, starts, stops):
        dphi = np.pi*np.ones_like(tphi, dtype=np.float32)
        for iev, (start, stop) in enumerate(zip(starts, stops)):
            for iob in range(start, stop):
                dphi[iob] = np.abs(BoundPhi(tphi[iob]-mephi[iev]))

        return dphi.astype(np.float32)

    tphi = ev.Tau.phi
    return awk.JaggedArray(
        tphi.starts, tphi.stops, dphi_met(
            getattr(ev, "{}_phiShift".format(coll))(ev, source, nsig),
            tphi.content, tphi.starts, tphi.stops,
        ),
    )

def muon_pt_shift(ev, source, nsig):
    @nb.njit(["float32[:](float32[:],float32[:])"])
    def nb_muon_pt_err_v2(muon_pt, muon_eta):
        muon_pt_err = np.zeros_like(muon_pt, dtype=np.float32)
        for idx in range(len(muon_pt)):
            # see https://twiki.cern.ch/twiki/bin/view/CMS/MuonReferenceScaleResolRun2#RefRun
            pt_err = 0.
            if abs(muon_eta[idx])<1.2:
                pt_err = 0.004
            elif abs(muon_eta[idx])<2.1:
                pt_err = 0.009
            elif muon_eta[idx]>=2.1:
                pt_err = 0.017
            else:
                pt_err = 0.027
            muon_pt_err[idx] = pt_err*muon_pt[idx]
        return muon_pt_err

    muon_pt_err = nb_muon_pt_err_v2(ev.Muon.pt.content, ev.Muon.eta.content)
    shift = ((source=="muonPtScale")*muon_pt_err/ev.Muon.pt.content).astype(np.float32)
    return awk.JaggedArray(ev.Muon.pt.starts, ev.Muon.pt.stops, pt_shift_numba(
        ev.Muon.pt.content, nsig, shift, -1.*shift
    ))

def ele_pt_shift(ev, source, nsig):
    shift = ((source=="eleEnergyScale")*ev.Electron_energyErr.content/(ev.Electron.pt.content*np.cosh(ev.Electron.eta.content))).astype(np.float32)
    return awk.JaggedArray(ev.Electron.pt.starts, ev.Electron.pt.stops, pt_shift_numba(
        ev.Electron.pt.content, nsig, shift, -shift
    ))

def photon_pt_shift(ev, source, nsig):
    shift = ((source=="photonEnergyScale")*ev.Photon_energyErr.content/(ev.Photon.pt.content*np.cosh(ev.Photon.eta.content))).astype(np.float32)
    result = awk.JaggedArray(ev.Photon.pt.starts, ev.Photon.pt.stops, pt_shift_numba(
        ev.Photon.pt.content, nsig, shift, -shift
    ))
    result.content[np.isnan(result).content] = 0.
    return result

def met_shift(ev, source, nsig, coll, attr):
    @nb.njit([
        "UniTuple(float32[:],2)("
        "float32[:],float32[:],"
        "float32[:],float32[:],float32[:],int64[:],int64[:],"
        ")"
    ])
    def met_shift_numba(met, mephi, opt, optcorr, ophi, ostarts, ostops):
        opx_old, opy_old = RadToCart2D(opt, ophi)
        opx_new, opy_new = RadToCart2D(optcorr, ophi)

        mex, mey = RadToCart2D(met, mephi)
        for iev, (osta, osto) in enumerate(zip(ostarts, ostops)):
            for iob in range(osta, osto):
                mex[iev] += (opx_old[iob] - opx_new[iob])
                mey[iev] += (opy_old[iob] - opy_new[iob])

        return CartToRad2D(mex, mey)

    arg_ = 1 if attr=='phi' else 0

    met = getattr(ev, "{}_pt".format(coll))
    mephi = getattr(ev, "{}_phi".format(coll))

    # Have to apply JEC shifts
    nmet, nmephi = met_shift_numba(
        met, mephi,
        ev.Jet_pt.content, ev.Jet_ptShift(ev, source, nsig).content,
        ev.Jet_phi.content, ev.Jet_pt.starts, ev.Jet_pt.stops,
    )
    if source in ["eleEnergyScale"]:
        nmet, nmephi = met_shift_numba(
            nmet, nmephi,
            ev.Electron_pt.content, ev.Electron_ptShift(ev, source, nsig).content,
            ev.Electron_phi.content, ev.Electron_phi.starts, ev.Electron_phi.stops,
        )
    elif source in ["muonPtScale"]:
        nmet, nmephi = met_shift_numba(
            nmet, nmephi,
            ev.Muon_pt.content, ev.Muon_ptShift(ev, source, nsig).content,
            ev.Muon_phi.content, ev.Muon_phi.starts, ev.Muon_phi.stops,
        )
    elif source in ["photonEnergyScale"]:
        nmet, nmephi = met_shift_numba(
            nmet, nmephi,
            ev.Photon_pt.content, ev.Photon_ptShift(ev, source, nsig).content,
            ev.Photon_phi.content, ev.Photon_phi.starts, ev.Photon_phi.stops,
        )
    elif source in ["tauPtScale"]:
        nmet, nmephi = met_shift_numba(
            nmet, nmephi,
            ev.Tau_pt.content, ev.Tau_ptShift(ev, source, nsig).content,
            ev.Tau_phi.content, ev.Tau_phi.starts, ev.Tau_phi.stops,
        )
    elif source in ["unclust"]:
        mex, mey = RadToCart2D(nmet, nmephi)
        mex += nsig*ev.MET_MetUnclustEnUpDeltaX
        mey += nsig*ev.MET_MetUnclustEnUpDeltaY
        nmet, nmephi = CartToRad2D(mex, mey)

    return (nmet, nmephi)[arg_].astype(np.float32)

def met_sumet_shift(ev, source, nsig, coll):
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
        getattr(ev, "{}_sumEt".format(coll)),
        ev.Jet_pt.content, ev.Jet_ptShift(ev, source, nsig).content,
        ev.Jet_eta.starts, ev.Jet_eta.stops,
        ev.Electron_pt.content, ev.Electron_ptShift(ev, source, nsig).content,
        ev.Electron_eta.starts, ev.Electron_eta.stops,
        ev.Muon_pt.content, ev.Muon_ptShift(ev, source, nsig).content,
        ev.Muon_eta.starts, ev.Muon_eta.stops,
        ev.Photon_pt.content, ev.Photon_ptShift(ev, source, nsig).content,
        ev.Photon_eta.starts, ev.Photon_eta.stops,
        ev.Tau_pt.content, ev.Tau_ptShift(ev, source, nsig).content,
        ev.Tau_eta.starts, ev.Tau_eta.stops,
    )

def obj_selection(ev, source, nsig, attr, name, sele, xclean=False):
    mask = getattr(ev, "{}_{}Mask".format(name, sele))(ev, source, nsig)
    if xclean:
        mask = mask & getattr(ev, "{}_XCleanMask".format(name))(ev, source, nsig)

    obj = getattr(ev, "{}_{}".format(name, attr))
    if callable(obj):
        obj = obj(ev, source, nsig)

    return obj[mask]

def obj_drtrig(ev, source, nsig, coll, ref, ref_selection=None):
    @nb.njit(["float32[:](int64[:], int64[:], float32[:], float32[:], int64[:], int64[:], float32[:], float32[:])"])
    def nb_dr_coll_ref(
        coll_starts, coll_stops, coll_eta, coll_phi,
        ref_starts, ref_stops, ref_eta, ref_phi,
    ):
        # maximally opposite in eta and phi
        coll_dr = (10.+np.pi)*np.ones_like(coll_eta, dtype=np.float32)

        for cstart, cstop, rstart, rstop in zip(
            coll_starts, coll_stops, ref_starts, ref_stops,
        ):
            for ic in range(cstart, cstop):
                coll_dr[ic] = min([
                    DeltaR2(
                        coll_eta[ic]-ref_eta[ir],
                        coll_phi[ic]-ref_phi[ir],
                    ) for ir in range(rstart, rstop)
                ]+[10.+np.pi])
        return coll_dr.astype(np.float32)

    ref_eta = getattr(ev, ref).eta
    if ref_selection is not None:
        mask = Lambda(ref_selection)(ev, source, nsig)
    else:
        mask = awk.JaggedArray(
            ref_eta.starts, ref_eta.stops,
            np.ones_like(ref_eta.content),
        )

    starts, stops = getattr(ev, coll).eta.starts, getattr(ev, coll).eta.stops
    return awk.JaggedArray(
        starts, stops,
        nb_dr_coll_ref(
            starts, stops,
            getattr(ev, coll).eta.content,
            getattr(ev, coll).phi.content,
            ref_eta[mask].starts, ref_eta[mask].stops,
            ref_eta[mask].content,
            getattr(ev, ref).phi[mask].content,
        ),
    )

def tau_pt_shift(ev, source, nsig):
    # see https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendation13TeV#Tau_energy_scale
    # corrections summed in quad to uncertainties are still dominated by
    # uncertainties, so lets use the quad
    @nb.njit([
        "float32[:](float32[:],float32[:])",
        "float32[:](float32[:],int32[:])",
    ])
    def nb_tau_pt_err(tau_pt, tau_dm):
        tau_pt_err = np.zeros_like(tau_pt, dtype=np.float32)
        for idx in range(len(tau_pt)):
            pt_err = 0.
            if (-0.5<tau_dm[idx]) and (tau_dm[idx]<0.5):
                pt_err = 0.012 #0.010
            elif (0.5<tau_dm[idx]) and (tau_dm[idx]<2.5):
                pt_err = 0.010 #0.009
            elif (9.5<tau_dm[idx]) and (tau_dm[idx]<10.5):
                pt_err = 0.011 #0.011
            tau_pt_err[idx] = pt_err*tau_pt[idx]
        return tau_pt_err

    tau_pt_err = nb_tau_pt_err(
        ev.Tau_pt.content, ev.Tau_decayMode.content,
    )
    shift = ((source=="tauPtScale")*tau_pt_err/ev.Tau_pt.content).astype(np.float32)
    return awk.JaggedArray(ev.Tau_pt.starts, ev.Tau_pt.stops, pt_shift_numba(
        ev.Tau_pt.content, nsig, shift, -1.*shift
    ))

class ObjectFunctions(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.register_function(event, "Jet_ptShift", jet_pt_shift)
        event.register_function(event, "Muon_ptShift", muon_pt_shift)
        event.register_function(event, "Electron_ptShift", ele_pt_shift)
        event.register_function(event, "Photon_ptShift", photon_pt_shift)
        event.register_function(event, "Tau_ptShift", tau_pt_shift)
        event.register_function(event, "MET_ptShift", partial(met_shift, coll="MET", attr='pt'))
        event.register_function(event, "MET_phiShift", partial(met_shift, coll="MET", attr='phi'))
        event.register_function(event, "MET_sumEtShift", partial(met_sumet_shift, coll="MET"))
        event.register_function(event, "PuppiMET_ptShift", partial(met_shift, coll="PuppiMET", attr='pt'))
        event.register_function(event, "PuppiMET_phiShift", partial(met_shift, coll="PuppiMET", attr='phi'))
        event.register_function(event, "PuppiMET_sumEtShift", partial(met_sumet_shift, coll="PuppiMET"))
        event.register_function(event, "Jet_dphiMET", partial(jet_dphimet, coll="MET"))
        event.register_function(event, "Jet_dphiPuppiMET", partial(jet_dphimet, coll="PuppiMET"))
        event.register_function(event, "Tau_dphiMET", partial(tau_dphimet, coll="MET"))
        event.register_function(event, "Tau_dphiPuppiMET", partial(tau_dphimet, coll="PuppiMET"))
        event.register_function(event, "Muon_dRTrigMuon", partial(
            obj_drtrig, coll="Muon", ref="TrigObj", ref_selection="ev, source, nsig: (np.abs(ev.TrigObj_id)==1) | (np.abs(ev.TrigObj_id)==13)",
        ))
        event.register_function(event, "Electron_dRTrigElectron", partial(
            obj_drtrig, coll="Electron", ref="TrigObj", ref_selection="ev, source, nsig: np.abs(ev.TrigObj_id)==11",
        ))

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
