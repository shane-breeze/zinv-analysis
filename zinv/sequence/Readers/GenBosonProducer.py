import awkward as awk
import numpy as np
import numba as nb
from functools import partial

from . import Collection
from zinv.utils.Geometry import DeltaR2, LorTHPMToXYZE, LorXYZEToTHPM

class GenBosonProducer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.register_function(event, "nGenBosons", ngen_bosons)
        event.register_function(event, "GenLepCandidates", genpart_candidates)
        event.register_function(event, "GenPartBoson", genpart_boson)

def ngen_bosons(ev):
    gp = ev.GenPart
    pdgs = gp.pdgId
    motheridx = gp.genPartIdxMother

    nb_mask = (
        (np.abs(pdgs)==11) | (np.abs(pdgs)==12) | (np.abs(pdgs)==13)
        | (np.abs(pdgs)==14) | (np.abs(pdgs)==15) | (np.abs(pdgs)==16)
        | (np.abs(pdgs)==23) | (np.abs(pdgs)==24)
    ) & (motheridx==0)

    nbosons = np.zeros_like(pdgs.content, dtype=np.int32)
    nbosons[(nb_mask & (np.abs(pdgs)<20)).content] = 1
    nbosons[(nb_mask & (np.abs(pdgs)>=20)).content] = 2
    nbosons = awk.JaggedArray(pdgs.starts, pdgs.stops, nbosons)
    return nbosons.sum()/2

def create_genpart_candidates(ev, gp_mask, gdl_idx):
    @nb.njit(["UniTuple(float32[:],5)(int32[:],float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],int64[:],int32[:],float32[:],float32[:],float32[:],float32[:],int64[:],int64[:])"])
    def create_genpart_candidates_jit(
        gps_pdgId, gps_pt, gps_eta, gps_phi, gps_mass, gps_gdidx, gps_starts, gps_stops,
        gds_pdgId, gds_pt, gds_eta, gds_phi, gds_mass, gds_starts, gds_stops,
    ):

        pdgs = np.zeros_like(gps_pt, dtype=np.float32)
        pts = np.zeros_like(gps_pt, dtype=np.float32)
        etas = np.zeros_like(gps_pt, dtype=np.float32)
        phis = np.zeros_like(gps_pt, dtype=np.float32)
        masss = np.zeros_like(gps_pt, dtype=np.float32)

        for iev, (gps_start, gps_stop, gds_start, gds_stop) in enumerate(zip(
            gps_starts, gps_stops, gds_starts, gds_stops,
        )):
            x, y, z, e = 0., 0., 0., 0.
            for igps in range(gps_start, gps_stop):
                igds = gps_gdidx[igps]
                if igds >= 0:
                    igds += gds_start
                    pdgs[igps] = gds_pdgId[igds]
                    pts[igps] = gds_pt[igds]
                    etas[igps] = gds_eta[igds]
                    phis[igps] = gds_phi[igds]
                    masss[igps] = gds_mass[igds]
                else:
                    pdgs[igps] = gps_pdgId[igps]
                    pts[igps] = gps_pt[igps]
                    etas[igps] = gps_eta[igps]
                    phis[igps] = gps_phi[igps]
                    masss[igps] = gps_mass[igps]
        return pdgs, pts, etas, phis, masss

    starts = ev.GenPart.pt[gp_mask].starts
    stops = ev.GenPart.pt[gp_mask].stops

    pdg, pt, eta, phi, mass = create_genpart_candidates_jit(
        ev.GenPart.pdgId[gp_mask].content,
        ev.GenPart.pt[gp_mask].content,
        ev.GenPart.eta[gp_mask].content,
        ev.GenPart.phi[gp_mask].content,
        ev.GenPart.mass[gp_mask].content,
        gdl_idx, starts, stops,
        ev.GenDressedLepton.pdgId.content,
        ev.GenDressedLepton.pt.content,
        ev.GenDressedLepton.eta.content,
        ev.GenDressedLepton.phi.content,
        ev.GenDressedLepton.mass.content,
        ev.GenDressedLepton.pt.starts,
        ev.GenDressedLepton.pt.stops,
    )

    return (
        awk.JaggedArray(starts, stops, pdg),
        awk.JaggedArray(starts, stops, pt),
        awk.JaggedArray(starts, stops, eta),
        awk.JaggedArray(starts, stops, phi),
        awk.JaggedArray(starts, stops, mass),
    )

def genpart_candidates(ev, attr):
    if not ev.hasbranch("GenLepCandidates_{}".format(attr)):
        gp = ev.GenPart
        flags = gp.statusFlags
        pdgs = gp.pdgId
        status = gp.status

        # Mask for boson decay products
        gp_mask = (
            ((flags&1==1) & (flags&(1<<8)==(1<<8)))
            & (
                (((np.abs(pdgs)==11) | (np.abs(pdgs)==13)) & (status==1) & ((flags&(1<<2)==0)))
                | ((np.abs(pdgs)==15) & (status==2))
                | (((np.abs(pdgs)==12) | (np.abs(pdgs)==14) | (np.abs(pdgs)==16)) & (status==1))
            )
        )

        genpart_dressedlepidx = genpart_matched_dressedlepton(ev, gp_mask)

        pdgId, pt, eta, phi, mass = create_genpart_candidates(
            ev, gp_mask, genpart_dressedlepidx,
        )
        ev.GenLepCandidates_pdgId = pdgId.astype(np.int32)
        ev.GenLepCandidates_pt = pt
        ev.GenLepCandidates_eta = eta
        ev.GenLepCandidates_phi = phi
        ev.GenLepCandidates_mass = mass

    return getattr(ev, "GenLepCandidates_{}".format(attr))

def genpart_boson(ev, attr):
    if not ev.hasbranch("GenPartBoson_{}".format(attr)):
        pt, eta, phi, mass = create_genpart_boson(ev)
        ev.GenPartBoson_pt = pt
        ev.GenPartBoson_eta = eta
        ev.GenPartBoson_phi = phi
        ev.GenPartBoson_mass = mass

    return getattr(ev, "GenPartBoson_{}".format(attr))

def create_genpart_boson(ev):
    @nb.njit(["UniTuple(float32[:],4)(float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],int64)"])
    def create_genpart_boson_jit(pt, eta, phi, mass, starts, stops, nev):
        opt = np.zeros(nev, dtype=np.float32)
        oeta = np.zeros(nev, dtype=np.float32)
        ophi = np.zeros(nev, dtype=np.float32)
        omass = np.zeros(nev, dtype=np.float32)

        for iev, (start, stop) in enumerate(zip(starts, stops)):
            x, y, z, e = 0., 0., 0., 0.
            for iob in range(start, stop):
                tx, ty, tz, te = LorTHPMToXYZE(
                    pt[iob], eta[iob], phi[iob], mass[iob],
                )
                x += tx
                y += ty
                z += tz
                e += te
            t, h, p, m = LorXYZEToTHPM(x, y, z, e)
            opt[iev] = t
            oeta[iev] = h
            ophi[iev] = p
            omass[iev] = m
        return opt, oeta, ophi, omass

    return create_genpart_boson_jit(
        ev.GenLepCandidates(ev, 'pt').content,
        ev.GenLepCandidates(ev, 'eta').content,
        ev.GenLepCandidates(ev, 'phi').content,
        ev.GenLepCandidates(ev, 'mass').content,
        ev.GenLepCandidates(ev, 'pt').starts,
        ev.GenLepCandidates(ev, 'pt').stops,
        ev.size,
    )

def genpart_matched_dressedlepton(ev, gpmask):
    @nb.njit(["int64[:](int32[:],float32[:],float32[:],int64[:],int64[:],int32[:],float32[:],float32[:],int64[:],int64[:])"])
    def genpart_matched_dressedlepton_jit(
        gps_pdg, gps_eta, gps_phi, gps_starts, gps_stops,
        gds_pdg, gds_eta, gds_phi, gds_starts, gds_stops,
    ):
        indices = -1*np.ones(gps_pdg.shape[0], dtype=np.int32)
        for iev, (gps_start, gps_stop, gds_start, gds_stop) in enumerate(zip(
            gps_starts, gps_stops, gds_starts, gds_stops,
        )):
            for igps in range(gps_start, gps_stop):
                for igds in range(gds_start, gds_stop):
                    matched_pdg = (gps_pdg[igps] == gds_pdg[igds])
                    within_dr = (DeltaR2(gps_eta[igps]-gds_eta[igds],
                                         gps_phi[igps]-gds_phi[igds]) < 0.01)
                    if matched_pdg and within_dr:
                        indices[igps] = igds - gds_start
                        break
        return indices

    return genpart_matched_dressedlepton_jit(
        ev.GenPart.pdgId[gpmask].content,
        ev.GenPart.eta[gpmask].content,
        ev.GenPart.phi[gpmask].content,
        ev.GenPart.pdgId[gpmask].starts,
        ev.GenPart.pdgId[gpmask].stops,
        ev.GenDressedLepton.pdgId.content,
        ev.GenDressedLepton.eta.content,
        ev.GenDressedLepton.phi.content,
        ev.GenDressedLepton.pdgId.starts,
        ev.GenDressedLepton.pdgId.stops,
    )
